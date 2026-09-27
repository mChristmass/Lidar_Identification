"""Matched 20-epoch continuation controls; NYUv2 train/dev only."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .corruptions import stable_seed
from .corruptions_v2 import corrupt, FAMILIES, RATES
from .data import read_manifest, robust_depth
from .diagnose_phase1 import DiagnosticData, digest, regional
from .model import CMXPaper
from .run import seed_all, save_json


def training_condition(seed, epoch, sample_id, regime):
    rng = np.random.default_rng(stable_seed(seed, 'train', str(sample_id), regime, str(epoch)))
    if regime == 'clean' or rng.random() < .5:
        return 'clean', 'medium'
    return FAMILIES[int(rng.integers(len(FAMILIES)))], tuple(RATES)[int(rng.integers(3))]


class TrainingData(Dataset):
    def __init__(self, root, regime, seed):
        self.root, self.regime, self.seed, self.epoch = root, regime, seed, 0
        self.rows = read_manifest(root / 'manifests/train.jsonl')
        dev = read_manifest(root / 'manifests/dev.jsonl')
        if len(self.rows) != 715 or len(dev) != 80:
            raise ValueError('Expected frozen tuning split 715/80')
        if {r['sample_id'] for r in self.rows} & {r['sample_id'] for r in dev}:
            raise ValueError('Train/dev overlap')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        a = np.asarray(Image.open(self.root / row['intensity']), np.float32)
        d = np.asarray(Image.open(self.root / row['depth']), np.float32)
        y = np.asarray(Image.open(self.root / row['label']), np.int64)
        family, severity = training_condition(self.seed, self.epoch, row['sample_id'], self.regime)
        a, d, valid, _ = corrupt(a, d, family, severity,
                                f'train/{self.epoch}/{row["sample_id"]}', self.seed)
        x = np.stack((a, robust_depth(d, valid), valid.astype(np.float32)))
        # Paired flip choices independent of training regime.
        rng = np.random.default_rng(stable_seed(self.seed, 'train', row['sample_id'], 'flip', str(self.epoch)))
        if rng.random() < .5:
            x, y = x[..., ::-1], y[:, ::-1]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def atomic_checkpoint(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_42/best.pth'))
    p.add_argument('--nyuv2-dir', type=Path, default=Path('data/public_semseg/nyuv2/processed'))
    p.add_argument('--output-dir', type=Path, default=Path('data/paper_benchmark_runs/mixed_control_v1'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum) < 1:
        raise ValueError('epochs/batch-size/grad-accum must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / 'RUNNING.lock'
    # Exclusive lock protects matched runs from concurrent overwrites. After a
    # hard kill, remove ONLY this lock after verifying that no job is running.
    with lock.open('x') as f:
        import os, socket
        json.dump({'pid': os.getpid(), 'host': socket.gethostname()}, f)
    try:
        execute(args)
    finally:
        lock.unlink(missing_ok=True)


def execute(args):
    fingerprint = digest(args.checkpoint)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(checkpoint_sha256=fingerprint, selection='fixed final epoch; no dev-based reselection',
                    augmentation='mixed: Bernoulli 0.5 clean, otherwise uniform family and severity, epoch-varying',
                    train_manifest_sha256=digest(args.nyuv2_dir / 'manifests/train.jsonl'),
                    dev_manifest_sha256=digest(args.nyuv2_dir / 'manifests/dev.jsonl'))
    protocol_path = args.output_dir / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Output protocol differs; use a new output directory')
    save_json(protocol_path, protocol)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = source['args']
    if (config['dataset'], config['model'], config['stage'], config['corruption']) != ('nyuv2', 'CMX_B2', 'tune', 'clean'):
        raise ValueError('Expected clean-trained NYUv2 B2 tuning checkpoint')
    for regime in ('clean', 'mixed'):
        destination = args.output_dir / regime
        destination.mkdir(exist_ok=True)
        seed_all(args.seed)
        model = CMXPaper('CMX_B2', 40, 640, config['decoder_dim']).to(args.device)
        model.load_state_dict(source['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        amp = str(args.device).startswith('cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        dataset = TrainingData(args.nyuv2_dir, regime, args.seed)
        start, history = 1, []
        last = destination / 'last.pth'
        if last.exists():
            state = torch.load(last, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            start, history = state['epoch'] + 1, state['history']
            del state
        for epoch in range(start, args.epochs + 1):
            seed_all(args.seed + epoch)
            dataset.epoch = epoch
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(args.seed + epoch))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for index, (x, y) in enumerate(loader):
                lr = args.lr * (1 - ((epoch - 1) + index / len(loader)) / args.epochs) ** .9
                for group in optimizer.param_groups:
                    group['lr'] = lr
                # Account for the final partial accumulation window by sample count.
                window_start = (index // args.grad_accum) * args.grad_accum * args.batch_size
                window_samples = min(args.grad_accum * args.batch_size, len(dataset) - window_start)
                x, y = x.to(args.device), y.to(args.device)
                with torch.autocast(device_type='cuda', enabled=amp):
                    ce = torch.nn.functional.cross_entropy(model(x)['logits'], y, ignore_index=255)
                    loss = ce * x.shape[0] / window_samples
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{regime} epoch={epoch} batch={index}')
                scaler.scale(loss).backward()
                if (index + 1) % args.grad_accum == 0 or index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum += float(ce.detach()) * x.shape[0]
            history.append({'epoch': epoch, 'loss': loss_sum / len(dataset)})
            atomic_checkpoint(last, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scaler': scaler.state_dict(), 'epoch': epoch, 'history': history})
            save_json(destination / 'history.json', {'epochs': history})
            print(f'CONTROL_EPOCH regime={regime} epoch={epoch}/{args.epochs} loss={history[-1]["loss"]:.6f}', flush=True)
        model.eval()
        conditions = [('clean', 'medium')] + [(f, s) for f in FAMILIES for s in RATES]
        scores = []
        for j, (family, severity) in enumerate(conditions, 1):
            result = regional(model, DiagnosticData(args.nyuv2_dir, family, severity), args.device)
            result.update(regime=regime, epoch=args.epochs, initialization_sha256=fingerprint)
            save_json(destination / 'curves' / f'{family}_{severity}.json', result)
            score = result['regions']['all']['miou']
            scores.append({'condition': f'{family}_{severity}', 'miou': score})
            print(f'CONTROL_EVAL regime={regime} condition={j}/22 {family}/{severity} miou={score:.6f}', flush=True)
        save_json(destination / 'result.json', {'scores': scores, 'protocol': protocol})
        del model, optimizer, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(args.output_dir / 'complete.json', {'training_runs': 2, 'epochs_each': args.epochs, 'evaluations': 44})
    print('MIXED_CONTROL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

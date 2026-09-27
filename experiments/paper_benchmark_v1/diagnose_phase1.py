"""Read-only checkpoint diagnosis. Writes only to a new diagnostic directory."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .corruptions_v2 import corrupt, FAMILIES, RATES
from .data import NYUv2PaperDataset, PrivatePaperDataset, robust_depth, read_manifest
from .model import CMXPaper
from .run import evaluate, save_json
from experiments.t8_transfer_statistics.model_transfer import T8TransferNet


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


class DiagnosticData(Dataset):
    def __init__(self, root, family, severity):
        self.root, self.family, self.severity = root, family, severity
        self.rows = read_manifest(root / 'manifests/dev.jsonl')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        a = np.asarray(Image.open(self.root / row['intensity']), np.float32)
        d = np.asarray(Image.open(self.root / row['depth']), np.float32)
        y = np.asarray(Image.open(self.root / row['label']), np.int64)
        original = np.isfinite(d) & (d > 0)
        a, d, valid, added = corrupt(a, d, self.family, self.severity, row['sample_id'])
        x = np.stack((a, robust_depth(d, valid), valid.astype(np.float32)))
        return {'inputs': torch.from_numpy(x.copy()), 'labels': torch.from_numpy(y.copy()),
                'original_valid': torch.from_numpy(original), 'added': torch.from_numpy(added),
                'sample_id': row['sample_id']}


@torch.no_grad()
def regional(model, dataset, device):
    matrices = {k: np.zeros((40, 40), np.int64) for k in
                ('all', 'natural_holes', 'added_holes', 'valid_depth', 'boundary')}
    audit = []
    for b in DataLoader(dataset, batch_size=1, num_workers=0):
        y = b['labels'][0].numpy()
        pred = model(b['inputs'].to(device))['logits'].argmax(1)[0].cpu().numpy()
        original = b['original_valid'][0].numpy()
        added = b['added'][0].numpy()
        valid = b['inputs'][0, 2].numpy() > 0
        boundary = np.zeros(y.shape, bool)
        ok = y != 255
        horizontal = (y[:, 1:] != y[:, :-1]) & ok[:, 1:] & ok[:, :-1]
        vertical = (y[1:] != y[:-1]) & ok[1:] & ok[:-1]
        boundary[:, 1:] |= horizontal
        boundary[:, :-1] |= horizontal
        boundary[1:] |= vertical
        boundary[:-1] |= vertical
        regions = dict(all=ok, natural_holes=~original, added_holes=added,
                       valid_depth=valid, boundary=boundary)
        for key, mask in regions.items():
            select = mask & ok
            matrices[key] += np.bincount(y[select] * 40 + pred[select], minlength=1600).reshape(40, 40)
        audit.append({'sample_id': b['sample_id'][0], 'natural_missing': float((~original).mean()),
                      'added_fraction_of_original_valid': float(added.sum() / max(original.sum(), 1)),
                      'total_missing': float((~valid).mean())})
    metrics = {}
    for key, matrix in matrices.items():
        union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
        iou = np.divide(matrix.diagonal(), union, out=np.full(40, np.nan), where=union > 0)
        metrics[key] = {'miou': float(np.nanmean(iou)) if (union > 0).any() else None,
                        'pixels': int(matrix.sum()), 'confusion_matrix': matrix.tolist()}
    return {'regions': metrics, 'degradation_audit': audit}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', type=Path, default=Path('data/paper_benchmark_runs/phase1_v1'))
    p.add_argument('--output-dir', type=Path, default=Path('data/paper_benchmark_runs/diagnosis_v2'))
    p.add_argument('--nyuv2-dir', type=Path, default=Path('data/public_semseg/nyuv2/processed'))
    p.add_argument('--private-dir', type=Path, default=Path('data/new_data/merged'))
    p.add_argument('--calibration', default='data/new_data/merged/paper_split_v1/degradation_profile_v1/calibration.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--audit-only', action='store_true', help='JSON history audit only; no inference')
    args = p.parse_args()
    paths = sorted(args.runs_dir.rglob('result.json'))
    if len(paths) != 36:
        raise RuntimeError(f'Expected 36 results, found {len(paths)}')
    records = []
    for path in paths:
        result = json.loads(path.read_text(encoding='utf-8'))
        history = json.loads((path.parent / 'history.json').read_text(encoding='utf-8'))['epochs']
        best = max(history, key=lambda row: row['dev_miou'])
        selected = next(row for row in history if row['epoch'] == result['best_epoch'])
        records.append({'path': path.parent.relative_to(args.runs_dir).as_posix(),
                        'reported': result, 'history_best': best,
                        'history_at_reported_epoch': selected,
                        'mismatch': abs(selected['dev_miou'] - result['best_selection_score']) > 1e-9})
    save_json(args.output_dir / 'history_audit.json', {'runs': records})
    print(f'HISTORY_AUDIT runs=36 mismatches={sum(r["mismatch"] for r in records)}', flush=True)
    if args.audit_only:
        return
    for i, (path, record) in enumerate(zip(paths, records), 1):
        checkpoint_path = path.parent / 'best.pth'
        fingerprint = digest(checkpoint_path)
        # Checkpoints are from the user's own trusted training run.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['args']
        classes = 2 if config['dataset'] == 'private' else 40
        if config['model'] == 'T8':
            model = T8TransferNet('T0_R1_REPRO', base_channels=config['t8_channels'], num_classes=classes)
        else:
            model = CMXPaper(config['model'], classes, max(config['height'], config['width']), config['decoder_dim'])
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint['model']
        model.to(args.device).eval()
        if classes == 2:
            dataset = PrivatePaperDataset(args.private_dir, args.private_dir / 'paper_split_v1/dev_indices.npy')
        else:
            dataset = NYUv2PaperDataset(args.nyuv2_dir, args.nyuv2_dir / 'manifests/dev.jsonl',
                       args.calibration, corruption=config['corruption'], severity=config['severity'],
                       split='dev', seed=config['seed'], image_size=(config['height'], config['width']))
        metrics = evaluate(model, DataLoader(dataset, batch_size=1, num_workers=0), args.device, classes)
        destination = args.output_dir / record['path']
        save_json(destination / 'reevaluation.json', {'checkpoint_sha256': fingerprint,
                  'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_args': config,
                  'metrics': metrics, 'history_mismatch': record['mismatch'],
                  'evaluation': 'FP32 dev only; historical v1 inputs; no checkpoint reselection'})
        print(f'REEVAL {i}/36 {record["path"]} miou={metrics["miou"]:.6f}', flush=True)
        # First diagnostic curve: one fixed clean-trained B2 seed; no retraining.
        if classes == 40 and config['model'] == 'CMX_B2' and config['seed'] == 42 and config['corruption'] == 'clean':
            conditions = [('clean', 'medium')] + [(f, s) for f in FAMILIES for s in RATES]
            for j, (family, severity) in enumerate(conditions, 1):
                out = regional(model, DiagnosticData(args.nyuv2_dir, family, severity), args.device)
                out.update(checkpoint_sha256=fingerprint, corruption=family, severity=severity,
                           protocol='v2 fixed diagnostic stress test; seed=20260926')
                save_json(args.output_dir / 'b2_clean_seed42_curves' / f'{family}_{severity}.json', out)
                print(f'CURVE {j}/22 {family}/{severity} miou={out["regions"]["all"]["miou"]}', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(args.output_dir / 'complete.json', {'reevaluations': 36, 'curve_conditions': 22})
    print('DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

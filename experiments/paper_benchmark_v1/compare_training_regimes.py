"""Evaluate existing joint-trained B2 on the frozen v2 dev suite."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .corruptions_v2 import FAMILIES, RATES
from .diagnose_phase1 import DiagnosticData, digest, regional
from .model import CMXPaper
from .run import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'data/paper_benchmark_runs/phase1_v1/nyuv2_joint_medium/nyuv2/tune/CMX_B2/seed_42/best.pth'))
    parser.add_argument('--clean-curves', type=Path, default=Path(
        'data/paper_benchmark_runs/diagnosis_v2/b2_clean_seed42_curves'))
    parser.add_argument('--nyuv2-dir', type=Path, default=Path('data/public_semseg/nyuv2/processed'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/paper_benchmark_runs/regime_comparison_v2'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    conditions = [('clean', 'medium')] + [(f, s) for f in FAMILIES for s in RATES]
    clean_results = {}
    for family, severity in conditions:
        name = f'{family}_{severity}'
        row = json.loads((args.clean_curves / f'{name}.json').read_text(encoding='utf-8'))
        if row['protocol'] != 'v2 fixed diagnostic stress test; seed=20260926':
            raise ValueError('Reference curves must use the identical v2 protocol')
        clean_results[name] = row
    fingerprint = digest(args.checkpoint)
    # Trusted checkpoints produced by this project, not third-party downloads.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['args']
    if (config['dataset'], config['model'], config['seed'], config['stage'], config['corruption'],
        config['severity'], config['height'], config['width']) != (
        'nyuv2', 'CMX_B2', 42, 'tune', 'joint', 'medium', 480, 640):
        raise ValueError('Expected phase1 NYUv2 joint-medium B2 seed42 tuning checkpoint')
    model = CMXPaper('CMX_B2', 40, 640, config['decoder_dim'])
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint['model']
    model.to(args.device).eval()
    summary = []
    for index, (family, severity) in enumerate(conditions, 1):
        name = f'{family}_{severity}'
        result = regional(model, DiagnosticData(args.nyuv2_dir, family, severity), args.device)
        reference = clean_results[name]
        # Check sample identities and actual degradation budgets before comparing.
        if result['degradation_audit'] != reference['degradation_audit']:
            raise RuntimeError(f'Dataset or degradation audit differs: {name}')
        result.update(checkpoint_sha256=fingerprint, checkpoint_args=config,
                      checkpoint_epoch=checkpoint['epoch'], corruption=family, severity=severity,
                      protocol=reference['protocol'])
        save_json(args.output_dir / 'joint_trained_curves' / f'{name}.json', result)
        clean_score = reference['regions']['all']['miou']
        joint_score = result['regions']['all']['miou']
        summary.append({'condition': name, 'clean_trained_miou': clean_score,
                        'joint_trained_miou': joint_score,
                        'gain_percentage_points': 100 * (joint_score - clean_score)})
        save_json(args.output_dir / 'comparison.json', {'completed': index, 'total': 22,
                  'checkpoint_sha256': fingerprint, 'rows': summary,
                  'scope': 'single-seed dev diagnostic; training uses v1, evaluation uses v2'})
        print(f'REGIME_COMPARE {index}/22 {name} clean_trained={clean_score:.6f} '
              f'joint_trained={joint_score:.6f} gain_pp={100*(joint_score-clean_score):.3f}', flush=True)
    print('REGIME_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

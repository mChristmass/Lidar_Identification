"""Inference-only modality stress test on three frozen NYUv2 dev conditions."""
from pathlib import Path

import torch

from .diagnose_phase1 import DiagnosticData, regional
from .model import CMXPaper
from .train_reliability import read, save

STRESS_CONDITIONS = (('clean', 'medium'), ('photon_proxy', 'heavy'), ('joint', 'heavy'))
MODES = ('original', 'intensity_zero', 'depth_zero')


class ModalityStress(DiagnosticData):
    def __init__(self, root, family, severity, mode):
        super().__init__(root, family, severity)
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.mode == 'intensity_zero':
            item['inputs'][0] = 0
        elif self.mode == 'depth_zero':
            item['inputs'][1:] = 0
        return item


@torch.no_grad()
def run_diagnostic(mixed_checkpoint, decoder_dim, root, output, reference, device, progress):
    """Returns after all nine inference evaluations, resuming saved conditions."""
    mixed_checkpoint = Path(mixed_checkpoint)
    output = Path(output)
    source = torch.load(mixed_checkpoint, map_location='cpu', weights_only=False)
    if source['epoch'] != 20:
        raise ValueError('Diagnostic checkpoint must be completed mixed R0 seed42')
    model = CMXPaper('CMX_B2', 40, 640, decoder_dim).to(device).eval()
    model.load_state_dict(source['model'], strict=True)
    del source
    rows = []
    for family, severity in STRESS_CONDITIONS:
        condition = f'{family}_{severity}'
        baseline = read(reference / f'{condition}.json')
        for mode in MODES:
            progress(phase='diagnostic', diagnostic_step=len(rows)+1,
                     diagnostic_total=9, condition=condition, modality=mode)
            path = output / f'{condition}_{mode}.json'
            if path.exists():
                row = read(path)
            else:
                row = regional(model, ModalityStress(root, family, severity, mode), device)
                row.update(condition=condition, mode=mode)
                save(path, row)
            if len(row['degradation_audit']) != len(baseline['degradation_audit']):
                raise ValueError(f'Sample count mismatch: {condition}/{mode}')
            # Original evaluation must reproduce the saved R0 curve; ablations
            # intentionally alter inputs and their missingness audit.
            if mode == 'original':
                if row['degradation_audit'] != baseline['degradation_audit']:
                    raise ValueError(f'Diagnostic source data changed: {condition}')
                if abs(row['regions']['all']['miou'] - baseline['regions']['all']['miou']) > 1e-5:
                    raise ValueError(f'R0 checkpoint/curve mismatch: {condition}')
            rows.append({'condition': condition, 'mode': mode,
                         'miou': row['regions']['all']['miou']})
            print(f'MODALITY_DIAGNOSTIC {len(rows)}/9 {condition} {mode} '
                  f'miou={rows[-1]["miou"]:.6f}', flush=True)
    save(output.parent / 'diagnostic_summary.json',
         {'interpretation': 'Input-removal stress test, not causal attribution or a deployable model',
          'rows': rows})
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

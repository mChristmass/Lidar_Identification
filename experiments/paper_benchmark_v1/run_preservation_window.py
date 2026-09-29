"""Three matched 20-epoch continuations from R0; seed42 development only."""
import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import run_noise_window as noise
from .diagnose_phase1 import digest
from .run_fusion_window import train_job
from .run_repair_window import checked_reference, choose
from .train_reliability import read, save


def preflight(args):
    prior = read(args.repair_dir/'protocol.json')
    checkpoint = args.reference_dir/'mixed/last.pth'
    if digest(checkpoint) != prior['references']['R0_seed42']['checkpoint']:
        raise ValueError('R0 differs from diagnosed reference')
    _, refs = checked_reference(args.reference_dir/'mixed')
    for split in ('train', 'dev'):
        if digest(args.nyuv2_dir/'manifests'/f'{split}.jsonl') != prior['manifests'][split]:
            raise ValueError('Frozen manifest changed')
    for key in refs:
        if digest(args.reference_dir/'mixed/curves'/f'{key}.json') != prior['references']['R0_seed42']['curves'][key]:
            raise ValueError('Reference curves changed')
    package = Path(__file__).parent
    files = sorted(package.glob('*.py')) + sorted((package.parent/'cmx_initial_transfer/official').glob('*.py'))
    protocol = dict(version=1, seed=42, epochs=20, initialization='same R0 epoch20 for all three groups',
                    checkpoint_sha256=prior['references']['R0_seed42']['checkpoint'],
                    manifests=prior['manifests'], reference_curves=prior['references']['R0_seed42']['curves'],
                    code={str(p.relative_to(package.parent)): digest(p) for p in files},
                    P0='R0 continued20 mixed CE; matched additional training budget',
                    P1='S2 new frontend + original losses + 1.0 KL to frozen R0 on intensity-inactive samples',
                    P2='S2 new frontend + original losses; segmenter parameters AND buffers frozen',
                    losses='CE+.2 paired KL+5 reconstruction+.1 gateBCE; P1 adds fixed-teacher KL(T=2)',
                    optimizer='AdamW base1e-5 added1e-3 wd.01 poly.9 clip1 batch2 accum4 workers0 AMP',
                    selection='fixed final epoch; no best-dev search; same original clean gate applied vs continued R0',
                    scope='single-seed dev screen, no final test, no private training')
    path = args.output_dir/'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Protocol changed; choose a new output directory')
    save(path, protocol)
    return protocol, refs


def execute(args):
    started = time.time()
    noise.update(args, phase='preflight')
    protocol, original_refs = preflight(args)
    state = torch.load(args.reference_dir/'mixed/last.pth', map_location='cpu', weights_only=False)
    if state['epoch'] != 20:
        raise ValueError('R0 must be epoch20')
    source = dict(model=state['model'], args={'decoder_dim': 256})
    del state
    args.window_progress = dict(window='preservation_v1', groups_completed=0, groups_possible=3, group_index=1)
    baseline = train_job(args, 'R0', 42, source, protocol, original_refs)
    baseline['comparison_role'] = 'P0 continued R0, not the original20 baseline'
    results = [baseline]
    save(args.output_dir/'comparison.json', dict(groups=results))
    refs = noise.references(args.output_dir/'R0_seed42/curves')
    candidates = []
    for index, (name, mode) in enumerate((('P1', 'teacher'), ('P2', 'frozen')), 2):
        args.window_progress = dict(window='preservation_v1', groups_completed=index-1,
                                    groups_possible=3, group_index=index)
        args.preservation, args.group_prefix = mode, name
        result = noise.train_candidate(args, 'S2', 42, source, protocol, refs)
        result.update(method=name, preservation=mode, comparison_role='gain vs P0 continued R0')
        result['gain_vs_original_R0_pp'] = {
            r['condition']: 100*(r['miou']-original_refs[r['condition']]['regions']['all']['miou'])
            for r in result['scores']}
        results.append(result)
        candidates.append(result)
        save(args.output_dir/'comparison.json', dict(groups=results))
        print(f'PRESERVATION_PROGRESS completed={index}/3 latest={name}', flush=True)
    selected = choose(candidates)
    summary = dict(groups=results, candidate=selected['method'] if selected else None,
                   gate='vs P0: clean>=-.15pp; 21-corruption mean>=.25pp; 6-noise mean>=.8pp',
                   note='Screen only, not statistical significance; inspect original-R0 gains too')
    save(args.output_dir/'comparison.json', summary)
    for result in results:
        clean_gain = next(r['gain_pp'] for r in result['scores'] if r['condition']=='clean_medium')
        print(f'PRESERVATION_SUMMARY group={result["group"]} clean={result["clean_miou"]:.6f} '
              f'corrupt_mean={result["corruption_mean_miou"]:.6f} clean_gain_pp={clean_gain:+.3f} '
              f'mean_gain_pp={result["corruption_mean_gain_pp"]:+.3f} role={result["comparison_role"]}', flush=True)
    save(args.output_dir/'complete.json', dict(training_runs=3, epochs_each=20, evaluations=66,
                                             elapsed_minutes=(time.time()-started)/60, candidate=summary['candidate']))
    args.window_progress = {}
    noise.update(args, phase='complete', groups_completed=3, groups_planned=3, candidate=summary['candidate'])
    (args.output_dir/'failure.json').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(reference_dir='data/paper_benchmark_runs/mixed_control_v1',
                    repair_dir='data/paper_benchmark_runs/repair_window_v1',
                    nyuv2_dir='data/public_semseg/nyuv2/processed',
                    output_dir='data/paper_benchmark_runs/preservation_window_v1')
    for key, value in defaults.items():
        parser.add_argument('--'+key.replace('_', '-'), type=Path, default=Path(value))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), handle)
    try:
        execute(args)
    except BaseException:
        error = traceback.format_exc()
        save(args.output_dir/'failure.json', dict(error=error))
        args.window_progress = {}
        noise.update(args, phase='failed', error=error)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

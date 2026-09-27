"""Read saved progress only. A stale timestamp is not a confirmed crash."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=Path('data/paper_benchmark_runs/reliability_pilot_v1'))
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    while True:
        path = args.runs_dir / 'status.json'
        state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'phase': 'not_started'}
        age = time.time() - state.get('updated_unix', time.time())
        print(time.strftime('%Y-%m-%d %H:%M:%S'),
              ' '.join(f'{k}={v}' for k, v in state.items() if k not in ('updated_unix', 'error')),
              f'age_seconds={age:.0f}', 'STALE_CHECK_PROCESS' if age > 1800 and state['phase'] not in ('complete', 'failed') else '',
              flush=True)
        if state.get('error'):
            print(state['error'], flush=True)
        if not args.watch or state['phase'] in ('complete', 'failed'):
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()

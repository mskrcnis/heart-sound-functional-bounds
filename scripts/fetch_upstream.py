"""Fetch the pinned Cambridge code; retain its academic noncommercial license."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/am2234/parallel-hsmm-murmur.git'
COMMIT = '5f9d14ee7f9215bf8cf523341241ee9d79be79ae'
TARGET = ROOT / 'phase9/third_party/parallel-hsmm-murmur'


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def verify(path):
    if not (path / '.git').is_dir():
        raise RuntimeError('Pinned upstream checkout missing; run python scripts/fetch_upstream.py')
    if git(path, 'rev-parse', 'HEAD') != COMMIT:
        raise RuntimeError('Unexpected upstream commit; existing checkout left unchanged')
    if git(path, 'status', '--porcelain'):
        raise RuntimeError('Upstream checkout has local changes; left unchanged')
    if not (path / 'LICENSE.md').is_file():
        raise RuntimeError('Upstream license is missing')
    print('Verified upstream commit ' + COMMIT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only or TARGET.exists():
        verify(TARGET)
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print('Fetching Cambridge code under GAP Available Source License v1.0.')
    subprocess.run(['git', 'init', str(TARGET)], check=True)
    subprocess.run(['git', '-C', str(TARGET), 'remote', 'add', 'origin', URL], check=True)
    subprocess.run(['git', '-C', str(TARGET), 'fetch', '--depth=1', 'origin', COMMIT], check=True)
    subprocess.run(['git', '-C', str(TARGET), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    verify(TARGET)


if __name__ == '__main__':
    main()

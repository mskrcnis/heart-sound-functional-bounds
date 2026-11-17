"""Regenerate the publication figures and tables in an isolated output folder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/reproduced')
    args = parser.parse_args()
    destination = args.output.resolve()
    build = (ROOT / 'build').resolve()
    if build not in destination.parents:
        parser.error('--output must be a new subdirectory of build/')
    if destination.exists():
        parser.error('Output already exists. Select a new --output below build/.')
    with tempfile.TemporaryDirectory(prefix='heart-sound-figures-') as temporary:
        stage = Path(temporary) / 'analysis'
        shutil.copytree(ROOT / 'analysis', stage)
        (stage / 'figures').mkdir(exist_ok=True)
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        subprocess.run([sys.executable, str(stage / 'scripts/prepare_evidence.py'), '--from-bundle'],
                       cwd=stage, env=env, check=True)
        compared = []
        paths = sorted((stage / 'tables').glob('*.tex')) + [
            stage / 'evidence/breadth_conditions.csv',
            stage / 'evidence/distillation_all_21_models.csv',
        ]
        for path in paths:
            relative = path.relative_to(stage)
            expected = ROOT / 'analysis' / relative
            if path.read_bytes() != expected.read_bytes():
                raise RuntimeError('Regenerated numeric output differs: ' + str(relative))
            compared.append(str(relative))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, destination)
        import matplotlib
        import numpy
        report = {
            'status': 'passed', 'table_and_csv_files_compared': compared,
            'figures': sorted(p.name for p in (destination / 'figures').iterdir()),
            'python': sys.version.split()[0], 'numpy': numpy.__version__,
            'matplotlib': matplotlib.__version__,
            'evidence_sha256': {p.name: digest(p) for p in sorted((ROOT / 'analysis/evidence').iterdir()) if p.is_file()},
            'scope': 'Aggregate figure/table reconstruction only; no model training or original witness rescoring.',
        }
        (destination / 'regeneration_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('Verified %d table/CSV files; figures written to %s' % (len(compared), destination.relative_to(ROOT)))


if __name__ == '__main__':
    main()

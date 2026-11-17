"""Run data-free numerical tests; optional pinned-upstream integration."""
import argparse
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--with-upstream', action='store_true', help='Run core and two synthetic upstream integration tests')
    group.add_argument('--upstream-only', action='store_true', help='Run only the two synthetic upstream integration tests')
    args = parser.parse_args()
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(ROOT / 'phase10/tests'))
    names = [] if args.upstream_only else [
        'test_path_bounds', 'test_cheap_queries', 'test_selection_aware',
        'test_query_gap_distillation',
    ]
    if args.with_upstream or args.upstream_only:
        from fetch_upstream import verify
        verify(ROOT / 'phase9/third_party/parallel-hsmm-murmur')
        names += [
            'test_query_training.TrainingChecks.test_recomputed_graph_torch_numpy_values_and_gradient',
            'test_query_training.TrainingChecks.test_metrics_identity_and_batch_padding',
        ]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())

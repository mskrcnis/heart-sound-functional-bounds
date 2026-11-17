#!/usr/bin/env python3
"""Run a deterministic, entirely synthetic three-frame bound example.

No patient recordings, trained models, or clinical labels are used. The values
illustrate a likelihood/confidence mismatch; they are not study results.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase10" / "src"))
from path_bounds import FunctionalBounds


def main():
    posterior = np.array([
        [0.56, 0.40],
        [0.56, 0.40],
        [0.40 ** 4 / 0.56 ** 3 * np.exp(-0.0001), 0.40],
    ], dtype=np.float64)
    duration = np.ones((1, 2), dtype=np.float64)
    transition = np.eye(2, dtype=np.float64)
    model = FunctionalBounds(posterior, duration, transition)
    result = model.interval(epsilon=0.0002)
    output = {
        "data": "synthetic; no patient data or trained model",
        "purpose": "Demonstrate model-conditional confidence bounds and feasible paths; not a diagnostic test.",
        "frames": len(posterior),
        "epsilon": result["epsilon"],
        "optimal_log_score": float(model.optimum),
        "nominal_path": model.base_path.tolist(),
        "nominal_confidence": float(model.base_confidence),
        "functional_interval": [result["lower"], result["upper"]],
        "frame_projection_interval": [result["projection_lower"], result["projection_upper"]],
        "feasible_confidence_range": [result["feasible_lower"], result["feasible_upper"]],
        "witnesses": {
            side: {"states": result["witness_" + side].tolist(),
                   "confidence": result["feasible_" + side],
                   "log_score": float(result["witness_" + side + "_log_score"])}
            for side in ("lower", "upper")
        },
        "solver_calls": result["solver_calls"],
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check release hashes and recompute saved-result arithmetic using the stdlib.

This is an aggregate consistency check. It does not rerun neural inference,
rescore path witnesses, or validate clinical performance. Historical witness
audits are retained separately under analysis/evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "analysis" / "evidence"
METHODS = ("frame", "segment", "functional", "intersect", "fixed")
MODELS = {"teacher_float", "teacher_int8"} | {
    f"{arm}_{seed}" for arm in ("hard", "kd")
    for seed in (20261107, 20261108, 20261109)
}
CONDITIONS = {(model, budget) for model in MODELS for budget in ("20", "full")}
GROUPS = {str(i) for i in range(1, 49)}
SEEDS = {20261113, 20261114, 20261115}
ETAS = ("0.001", "0.005", "0.01", "0.05")


class VerificationError(Exception):
    """A release check failed."""


def require(condition, message):
    # Deliberately avoid assert: checks also run with python -O.
    if not condition:
        raise VerificationError(message)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)),
            f"{label}: nonfinite number")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
            f"{label}: numeric mismatch")


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(name):
    with (EVIDENCE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def condition_key(row):
    return row["model"], str(row["budget"])


def indexed_conditions(rows, label):
    result = {condition_key(row): row for row in rows}
    require(len(result) == len(rows) == 16 and set(result) == CONDITIONS,
            f"{label}: expected exactly the 16 frozen model/budget conditions")
    return result


def verify_integrity():
    manifest_path = ROOT / "provenance" / "release_manifest.json"
    require(manifest_path.is_file(), "Missing provenance/release_manifest.json")
    files = read_json(manifest_path)["files"]
    require(isinstance(files, dict) and bool(files), "Manifest has no file hashes")
    for relative, expected in files.items():
        rel = PurePosixPath(relative)
        require(relative == str(rel) and not rel.is_absolute()
                and ".." not in rel.parts and "\\" not in relative,
                "Manifest contains an unsafe or noncanonical relative path")
        require(relative != "provenance/release_manifest.json", "Manifest cannot hash itself")
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                f"Invalid SHA-256 for {relative}")
        path = ROOT.joinpath(*rel.parts)
        require(path.is_file() and not path.is_symlink(), f"Missing regular file: {relative}")
        require(ROOT in path.resolve().parents, f"File escapes release: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == expected, f"SHA-256 mismatch: {relative}")
    return {"status": "passed", "files_checked": len(files)}


def sign(interval):
    low, high = interval["lower"], interval["upper"]
    require(math.isfinite(low) and math.isfinite(high) and low <= high,
            "Invalid interval endpoints")
    return "positive" if low > 0 else "nonpositive" if high <= 0 else "unresolved"


def width(row, method):
    return row[method]["upper"] - row[method]["lower"]


def verify_breadth():
    rows = read_json(EVIDENCE / "breadth_record_bounds_anonymized.json")
    summary = read_json(EVIDENCE / "functional_breadth_v1__summary.json")
    tightness = read_json(EVIDENCE / "functional_breadth_v1__tightness_diagnostic.json")
    require(len(rows) == 768, "Breadth evidence must contain 768 record-conditions")
    expected_fields = {
        "model", "budget", "number", "frames", "epsilon", "group_id", *METHODS,
        "feasible", "outer_possible_winners", "witnessed_winners",
        "stable_sign_multiple_winners", "feasible_both_signs", "seconds",
    }
    grouped = defaultdict(list)
    topologies = {"Holosystolic", "Early-systolic", "Mid-systolic"}
    for row in rows:
        require(set(row) == expected_fields, "Unexpected row fields in anonymized breadth evidence")
        require(row["group_id"] in GROUPS, "Group codes must be anonymous integers from 1 to 48")
        require(type(row["frames"]) is int and row["frames"] > 0, "Invalid frame count")
        close(row["epsilon"], 0.005 * row["frames"], "Breadth tolerance")
        for method in (*METHODS, "feasible"):
            require(set(row[method]) == {"lower", "upper", "sign"}, "Unexpected interval fields")
            require(sign(row[method]) == row[method]["sign"], f"{method}: sign mismatch")
        for field in ("witnessed_winners", "outer_possible_winners"):
            require(bool(row[field]) and set(row[field]) <= topologies
                    and len(set(row[field])) == len(row[field]), "Invalid topology winner list")
        require(set(row["witnessed_winners"]) <= set(row["outer_possible_winners"]),
                "Witnessed winner excluded from possible winners")
        for method in METHODS:
            require(row[method]["lower"] <= row["feasible"]["lower"] + 1e-10
                    and row[method]["upper"] >= row["feasible"]["upper"] - 1e-10,
                    f"{method}: outer interval excludes saved feasible endpoints")
        stable = sign(row["intersect"]) != "unresolved"
        both = row["feasible"]["lower"] <= 0 < row["feasible"]["upper"]
        require(type(row["feasible_both_signs"]) is bool and row["feasible_both_signs"] == both,
                "Both-sign witness flag mismatch")
        require(type(row["stable_sign_multiple_winners"]) is bool
                and row["stable_sign_multiple_winners"] == (stable and len(row["witnessed_winners"]) > 1),
                "Stable-sign/multiple-winner flag mismatch")
        require(not (stable and both), "Resolved sign conflicts with feasible both-sign interval")
        grouped[condition_key(row)].append(row)
    require(set(grouped) == CONDITIONS, "Breadth condition set mismatch")
    conditions = indexed_conditions(summary["conditions"], "Breadth summary")
    diagnostics = indexed_conditions(tightness["conditions"], "Tightness diagnostic")
    csv_rows = indexed_conditions(read_csv("breadth_conditions.csv"), "Breadth CSV")
    passed = 0
    for key, records in grouped.items():
        require(len(records) == 48 and {r["group_id"] for r in records} == GROUPS,
                "Each condition must contain the same 48 distinct anonymous groups")
        require({r["number"] for r in records} == set(range(48)), "Record indices must span 0 to 47")
        condition, diagnostic, flat = conditions[key], diagnostics[key], csv_rows[key]
        require(condition["n"] == diagnostic["records"] == int(flat["people"]) == 48,
                "Condition denominator mismatch")
        for method in METHODS:
            resolved = sum(sign(r[method]) != "unresolved" for r in records)
            require(resolved == condition["resolved"][method] == int(flat[method + "_resolved"]),
                    f"{method}: resolved count mismatch")
            close(statistics.median(width(r, method) for r in records),
                  condition["median_width"][method], f"{method}: median width")
        ratios = [width(r, "intersect") / width(r, "segment") for r in records
                  if width(r, "segment") > 1e-9]
        require(len(ratios) == condition["width_ratio_denominator"] == 48,
                "Paired width denominator mismatch")
        gain = 1 - statistics.median(ratios)
        close(gain, condition["median_paired_width_reduction_vs_segment"], "Paired width reduction")
        close(gain, flat["width_reduction"], "Paired width CSV")
        extra = sum(sign(r["segment"]) == "unresolved" and sign(r["intersect"]) != "unresolved"
                    for r in records)
        require(extra == condition["additional_resolved_vs_segment"], "Additional signs mismatch")
        gate = gain >= 0.5 and extra >= 5
        require(gate == condition["condition_gate_passed"], "Breadth condition gate mismatch")
        passed += gate
        counts = {
            "witnessed_both_signs": sum(r["feasible_both_signs"] for r in records),
            "witnessed_multiple_winners": sum(len(r["witnessed_winners"]) > 1 for r in records),
            "stable_sign_multiple_winners": sum(r["stable_sign_multiple_winners"] for r in records),
        }
        for label, value in counts.items():
            require(value == condition[label], f"{label}: aggregate mismatch")
            if label in flat:
                require(value == int(flat[label]), f"{label}: CSV mismatch")
        resolved = sum(sign(r["intersect"]) != "unresolved" for r in records)
        both = counts["witnessed_both_signs"]
        require((resolved, both, 48 - resolved - both) ==
                (diagnostic["resolved"], diagnostic["proved_both_signs"], diagnostic["remaining_undetermined"]),
                "Tightness sign partition mismatch")
        gaps = [width(r, "intersect") - width(r, "feasible") for r in records]
        close(statistics.median(gaps), diagnostic["median_outer_inner_gap"], "Median outer-inner gap")
        close(max(gaps), diagnostic["max_outer_inner_gap"], "Maximum outer-inner gap")
    resolved = sum(sign(r["intersect"]) != "unresolved" for r in rows)
    both = sum(r["feasible_both_signs"] for r in rows)
    remaining = len(rows) - resolved - both
    require((resolved, both, remaining) == (701, 48, 19), "Breadth sign partition differs from release")
    require(remaining == tightness["total_remaining_undetermined"], "Undetermined total mismatch")
    require(summary["record_conditions"] == 768 and summary["distinct_people"] == 48,
            "Summary denominator mismatch")
    require(passed == summary["conditions_passing"] == 16 and (passed >= 12) == summary["engineering_gate_passed"],
            "Breadth engineering gate mismatch")
    return {"status": "passed", "record_conditions": len(rows), "distinct_people": 48,
            "conditions": 16, "resolved_signs": resolved, "feasible_both_signs": both,
            "undetermined": remaining, "conditions_passing": passed}


def verify_distillation():
    csv_rows = read_csv("distillation_all_21_models.csv")
    require(len(csv_rows) == 21, "Expected all 21 distillation fits")
    indexed = {(r["trial"], int(r["seed"]), r["arm"]): r for r in csv_rows}
    require(len(indexed) == 21, "Duplicate distillation fit")
    results = {}
    for trial, filename, controls in (
        ("Fresh", "query_gap_training_v1__summary.json", ("kl", "score")),
        ("Warm", "query_gap_warmstart_v1__summary.json", ("kl", "score", "confidence")),
    ):
        summary = read_json(EVIDENCE / filename)
        arms = (*controls, "query")
        fits = summary["all_arms"]
        expected = {(seed, arm) for seed in SEEDS for arm in arms}
        require(len(fits) == len(expected) and {(r["seed"], r["arm"]) for r in fits} == expected,
                f"{trial}: incomplete or duplicate seed/arm set")
        require({(seed, arm) for label, seed, arm in indexed if label == trial} == expected,
                f"{trial}: CSV seed/arm set mismatch")
        fields = ("emission_kl", "score_mae", "global_bound_mae", "window_bound_mae",
                  "primary_bound_mae", "support_gap_mae") + (("response_mae",) if trial == "Warm" else ())
        by_seed = defaultdict(dict)
        for fit in fits:
            metrics = fit["metrics"]
            flat = indexed[trial, fit["seed"], fit["arm"]]
            require(metrics["n"] == int(flat["people"]) == 24, f"{trial}: fit denominator mismatch")
            for field in fields:
                close(metrics[field], flat[field], f"{trial}: {field} CSV")
                require(float(flat[field]) >= 0, f"{trial}: negative error metric")
            if trial == "Fresh":
                require(flat["response_mae"] == "", "Fresh response endpoint must remain unmeasured")
            close(metrics["primary_bound_mae"],
                  (metrics["global_bound_mae"] + metrics["window_bound_mae"]) / 2,
                  f"{trial}: primary endpoint")
            for eta in ETAS:
                close(metrics["sign_agreement"][eta], flat["agreement_" + eta], f"{trial}: sign agreement")
                require(0 <= metrics["sign_agreement"][eta] <= 1, "Agreement is outside [0, 1]")
            by_seed[fit["seed"]][fit["arm"]] = float(flat["primary_bound_mae"])
            close(metrics["primary_bound_mae"], summary["per_seed"][str(fit["seed"])][fit["arm"]],
                  f"{trial}: per-seed summary")
        means = {arm: {field: statistics.mean(float(indexed[trial, seed, arm][field]) for seed in SEEDS)
                       for field in fields} for arm in arms}
        for arm in arms:
            for field in fields:
                close(means[arm][field], summary["means"][arm][field], f"{trial}: {arm}/{field} mean")
        improvements = {arm: 1 - means["query"]["primary_bound_mae"] / means[arm]["primary_bound_mae"]
                        for arm in controls}
        for arm, value in improvements.items():
            close(value, summary["relative_primary_improvement"][arm], f"{trial}: relative improvement")
        wins = sum(all(by_seed[seed]["query"] < by_seed[seed][arm] for arm in controls) for seed in SEEDS)
        ratio = means["query"]["emission_kl"] / means["kl"]["emission_kl"]
        gate = all(value >= 0.1 for value in improvements.values()) and wins >= 2 and ratio <= 1.05
        if trial == "Warm":
            response_gain = 1 - means["query"]["response_mae"] / means["confidence"]["response_mae"]
            close(response_gain, summary["relative_response_improvement_vs_confidence"], "Warm response gain")
            close(ratio, summary["emission_kl_ratio"], "Warm emission ratio")
            require(wins == summary["paired_seed_wins"], "Warm paired wins mismatch")
            gate = gate and response_gain >= 0.1
        else:
            close(ratio, summary["query_to_kl_emission_error_ratio"], "Fresh emission ratio")
            require(wins == summary["paired_seeds_better_than_both"], "Fresh paired wins mismatch")
        require(gate == summary["engineering_gate_passed"] and gate is False,
                f"{trial}: expected failed engineering gate")
        results[trial.lower()] = {"fits": len(fits), "engineering_gate_passed": gate,
                                  "paired_seed_wins": wins, "relative_primary_improvement": improvements}
    return {"status": "passed", "fits": 21, "trials": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-integrity", action="store_true",
                        help="development only: skip release hashes while retaining all arithmetic checks")
    args = parser.parse_args()
    try:
        integrity = {"status": "skipped", "reason": "--skip-integrity"} if args.skip_integrity else verify_integrity()
        report = {"status": "passed", "integrity": integrity,
                  "breadth": verify_breadth(), "distillation": verify_distillation(),
                  "scope": "Release integrity and saved aggregate arithmetic; no new inference or clinical validation."}
    except (VerificationError, OSError, ValueError, TypeError, KeyError, IndexError) as error:
        message = str(error) if isinstance(error, VerificationError) else f"Malformed or unreadable evidence ({type(error).__name__})"
        print(json.dumps({"status": "failed", "error": message}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

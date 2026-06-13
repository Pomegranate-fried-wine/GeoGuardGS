import argparse
import csv
import json
import os
from collections import Counter, defaultdict


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def classify(item, args):
    tag = item.get("tag", "")
    confidence = float(item.get("confidence", 0.0) or 0.0)
    support = int(item.get("support_pixel_count", 0) or 0)
    mean_talpha = float(item.get("mean_talpha", 0.0) or 0.0)
    rgb_delta = float(item.get("rgb_patch_mae_delta", 0.0) or 0.0)
    da3_delta = float(item.get("da3_structure_delta", 0.0) or 0.0)
    border = bool(item.get("border_suspect", False))
    flags = item.get("evidence_flags", [])
    if isinstance(flags, str):
        flags = [flags] if flags and flags != "[]" else []

    protected = tag == "good_contributor" or rgb_delta > args.max_rgb_worsen
    low_evidence = tag == "low_evidence" or support < args.min_support_pixels or mean_talpha < args.min_mean_weight or border
    strong_bad = tag == "bad_contributor" and confidence >= args.min_confidence and da3_delta > args.min_da3_improvement and not protected and not low_evidence

    if protected:
        return "protect_candidate", ["good_or_rgb_sensitive"]
    if strong_bad and confidence >= args.prune_confidence:
        return "prune_candidate", ["strong_counterfactual_bad", "future_only_not_applied"]
    if strong_bad:
        return "opacity_decay_candidate", ["counterfactual_bad", "future_only_not_applied"]
    if low_evidence:
        return "skip", ["low_evidence"] + list(flags)
    return "skip", ["neutral_or_weak_counterfactual"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterfactual-dir", default="output/local_feedback/da3_structure_counterfactual_dryrun_A5000_top30")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--prune-confidence", type=float, default=0.9)
    parser.add_argument("--min-support-pixels", type=int, default=10)
    parser.add_argument("--min-mean-weight", type=float, default=0.01)
    parser.add_argument("--min-da3-improvement", type=float, default=0.01)
    parser.add_argument("--max-rgb-worsen", type=float, default=0.02)
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(args.counterfactual_dir, "pruning_candidate_tags")
    os.makedirs(args.output_dir, exist_ok=True)

    candidates_path = os.path.join(args.counterfactual_dir, "counterfactual_candidates.json")
    summary_path = os.path.join(args.counterfactual_dir, "counterfactual_summary.json")
    candidates = read_json(candidates_path)
    source_summary = read_json(summary_path) if os.path.exists(summary_path) else {}

    rows = []
    grouped = defaultdict(list)
    for item in candidates:
        action, reasons = classify(item, args)
        row = {
            "stable_gaussian_id": item.get("stable_gaussian_id"),
            "view_id": item.get("view_id"),
            "region_id": item.get("region_id"),
            "counterfactual_tag": item.get("tag"),
            "candidate_action": action,
            "confidence": item.get("confidence"),
            "support_pixel_count": item.get("support_pixel_count"),
            "mean_talpha": item.get("mean_talpha"),
            "da3_structure_delta": item.get("da3_structure_delta"),
            "rgb_patch_mae_delta": item.get("rgb_patch_mae_delta"),
            "border_suspect": item.get("border_suspect"),
            "reasons": reasons,
            "overlay_path": item.get("overlay_path"),
        }
        rows.append(row)
        grouped[str(item.get("stable_gaussian_id"))].append(row)

    counts = Counter(r["candidate_action"] for r in rows)
    summary = {
        "source_counterfactual_dir": args.counterfactual_dir,
        "counterfactual_objective": source_summary.get("counterfactual_objective", "da3_structure"),
        "uses_lidar_for_labeling": False,
        "gaussian_parameters_modified": False,
        "checkpoint_modified": False,
        "implemented_actions": "tag_only",
        "future_prune_is_not_executed": True,
        "thresholds": {
            "min_confidence": args.min_confidence,
            "prune_confidence": args.prune_confidence,
            "min_support_pixels": args.min_support_pixels,
            "min_mean_weight": args.min_mean_weight,
            "min_da3_improvement": args.min_da3_improvement,
            "max_rgb_worsen": args.max_rgb_worsen,
        },
        "counts": dict(counts),
        "candidate_count": len(rows),
    }

    write_json(os.path.join(args.output_dir, "pruning_candidate_summary.json"), summary)
    write_json(os.path.join(args.output_dir, "pruning_candidates.json"), rows)
    with open(os.path.join(args.output_dir, "pruning_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        fields = [
            "stable_gaussian_id", "view_id", "region_id", "counterfactual_tag", "candidate_action",
            "confidence", "support_pixel_count", "mean_talpha", "da3_structure_delta",
            "rgb_patch_mae_delta", "border_suspect", "reasons", "overlay_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

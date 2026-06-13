import csv
import json
import os
from collections import defaultdict

from lib.utils.cuda_contribution_utils import build_stable_id_map


BAD_TO_ACTION = {
    "bad_boundary_mixing_group": "shrink_candidate",
    "bad_edge_blurring_group": "opacity_decay_candidate",
    "bad_ranking_conflict_group": "split_candidate",
}

SUPPORTED_DRYRUN_ACTIONS = {
    "prune_candidate",
    "shrink_candidate",
    "split_candidate",
    "opacity_decay_candidate",
    "surface_align_candidate",
}


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class GaussianRepairOperator:
    """Tag-only repair operator for Gaussian groups.

    This class intentionally does not implement prune, shrink, split, or surface
    alignment. It converts group evidence into dry-run operation candidates and
    writes auditable manifests so the periodic controller can exercise the full
    pipeline before any structural operation is enabled.
    """

    def __init__(self, cfg_node):
        self.cfg = cfg_node
        self.enabled = bool(getattr(cfg_node, "enabled", False))
        self.control_mode = str(getattr(cfg_node, "control_mode", "off"))
        self.allow_parameter_modification = bool(getattr(cfg_node, "allow_parameter_modification", False))
        self.allow_real_prune = bool(getattr(cfg_node, "allow_real_prune", False))
        self.allow_real_split = bool(getattr(cfg_node, "allow_real_split", False))
        self.allow_real_shrink = bool(getattr(cfg_node, "allow_real_shrink", False))
        self.max_candidates = int(getattr(cfg_node, "max_controlled_gaussians", 256))
        self.min_confidence = float(getattr(cfg_node, "min_group_confidence", 0.0))
        self.min_support_pixels = int(getattr(cfg_node, "min_support_pixels", 8))
        self.skip_low_evidence = bool(getattr(cfg_node, "skip_low_evidence", True))
        self.rgb_safety_threshold = float(getattr(cfg_node, "rgb_safety_threshold", 0.02))

    def validate_no_real_structure_apply(self):
        if self.control_mode in {"prune_apply", "shrink_apply", "split_apply", "repair_apply"}:
            raise RuntimeError("Real prune/shrink/split is not enabled in this stage.")
        if self.allow_real_prune or self.allow_real_split or self.allow_real_shrink:
            raise RuntimeError("Real prune/shrink/split is not enabled in this stage.")

    def run_dryrun(self, model, group_evidence, protected_ids, output_dir, iteration=None):
        self.validate_no_real_structure_apply()
        os.makedirs(output_dir, exist_ok=True)
        stable_lookup, total_gaussians = self._stable_lookup(model)
        rows = []
        skipped = []
        protected_ids = {int(v) for v in protected_ids}
        for ev in group_evidence:
            action = self._infer_action(ev)
            if action not in SUPPORTED_DRYRUN_ACTIONS:
                skipped.append(self._skip_row(ev, "unsupported_or_skip_action"))
                continue
            if self._is_low_evidence(ev):
                skipped.append(self._skip_row(ev, "low_evidence"))
                continue
            if float(ev.get("confidence", 0.0)) < self.min_confidence:
                skipped.append(self._skip_row(ev, "confidence_below_threshold"))
                continue
            if float(ev.get("rgb_delta", 0.0)) > self.rgb_safety_threshold:
                skipped.append(self._skip_row(ev, "rgb_risk_above_threshold"))
                continue
            for gid in ev.get("stable_gaussian_ids", []):
                gid = int(gid)
                if gid in protected_ids:
                    skipped.append(self._skip_row(ev, "protected", gid))
                    continue
                meta = stable_lookup.get(gid, {})
                row = {
                    "iteration": int(iteration) if iteration is not None else "",
                    "operation": action,
                    "dryrun_only": True,
                    "will_modify_parameters": False,
                    "stable_gaussian_id": gid,
                    "view_local_id": meta.get("row_id", ""),
                    "model_name": meta.get("model_name", ""),
                    "model_local_id": meta.get("local_id", ""),
                    "group_id": ev.get("group_id", ""),
                    "group_label": ev.get("group_label", ""),
                    "future_action": ev.get("future_action", ""),
                    "confidence": ev.get("confidence", 0.0),
                    "support_pixels": ev.get("support_pixels", 0),
                    "risk_weighted_contribution": ev.get("risk_weighted_contribution", 0.0),
                    "mean_talpha": ev.get("mean_talpha", 0.0),
                    "max_talpha": ev.get("max_talpha", 0.0),
                    "risk_source": ev.get("risk_source", ""),
                    "supervision_mode": ev.get("supervision_mode", ""),
                    "counterfactual_objective": ev.get("counterfactual_objective", ""),
                    "is_protected": False,
                    "is_low_evidence": False,
                    "rgb_delta": ev.get("rgb_delta", 0.0),
                    "reason": self._action_reason(action, ev),
                }
                rows.append(row)
                if len(rows) >= self.max_candidates:
                    break
            if len(rows) >= self.max_candidates:
                break

        counts = defaultdict(int)
        for row in rows:
            counts[row["operation"]] += 1
        manifest = {
            "iteration": int(iteration) if iteration is not None else None,
            "status": "valid",
            "control_mode": self.control_mode,
            "candidate_count": len(rows),
            "skipped_count": len(skipped),
            "operation_counts": dict(counts),
            "total_gaussians": total_gaussians,
            "gaussian_parameters_modified": False,
            "real_prune_enabled": False,
            "real_split_enabled": False,
            "real_shrink_enabled": False,
            "note": "prune/shrink/split are dry-run tags only; no Gaussian structure is changed.",
        }
        _write_json(os.path.join(output_dir, "repair_operator_manifest.json"), manifest)
        _write_csv(os.path.join(output_dir, "repair_dryrun_candidates.csv"), rows)
        _write_csv(os.path.join(output_dir, "repair_dryrun_skipped.csv"), skipped)
        _write_json(os.path.join(output_dir, "repair_operator_audit.json"), {
            "status": "passed",
            "candidate_count": len(rows),
            "checks": {
                "no_real_prune": True,
                "no_real_split": True,
                "no_real_shrink": True,
                "no_parameter_modification": True,
                "protected_gaussians_skipped": True,
                "low_evidence_skipped": True,
            },
        })
        return manifest

    def _stable_lookup(self, model):
        stable_ids, model_names, model_local = build_stable_id_map(model)
        lookup = {}
        for row, gid in enumerate(stable_ids):
            lookup[int(gid)] = {
                "row_id": int(row),
                "model_name": str(model_names[row]),
                "local_id": int(model_local[row]),
            }
        return lookup, int(len(stable_ids))

    def _infer_action(self, ev):
        action = str(ev.get("future_action", "skip"))
        if action in SUPPORTED_DRYRUN_ACTIONS:
            return action
        label = str(ev.get("group_label", ""))
        return BAD_TO_ACTION.get(label, "skip")

    def _is_low_evidence(self, ev):
        if not self.skip_low_evidence:
            return False
        return bool(ev.get("is_low_evidence", False)) or int(ev.get("support_pixels", 0) or 0) < self.min_support_pixels

    def _skip_row(self, ev, reason, gid=""):
        return {
            "stable_gaussian_id": gid,
            "group_id": ev.get("group_id", ""),
            "group_label": ev.get("group_label", ""),
            "future_action": ev.get("future_action", ""),
            "confidence": ev.get("confidence", 0.0),
            "support_pixels": ev.get("support_pixels", 0),
            "skip_reason": reason,
            "will_modify_parameters": False,
        }

    def _action_reason(self, action, ev):
        label = ev.get("group_label", "")
        if action == "shrink_candidate":
            return f"{label}: possible over-coverage near boundary; dry-run only"
        if action == "split_candidate":
            return f"{label}: possible depth/ranking conflict; dry-run only"
        if action == "prune_candidate":
            return f"{label}: strong negative evidence required before future prune; dry-run only"
        if action == "opacity_decay_candidate":
            return f"{label}: conservative opacity control candidate; dry-run only"
        if action == "surface_align_candidate":
            return f"{label}: possible thin/surface alignment candidate; dry-run only"
        return "dry-run only"

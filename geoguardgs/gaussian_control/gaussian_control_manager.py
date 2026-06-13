import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from lib.utils.cuda_contribution_utils import build_stable_id_map
from lib.utils.general_utils import inverse_sigmoid


PROTECT_LABELS = {"good_boundary_support_group", "rgb_protect_group", "protect", "protect_candidate"}
BAD_LABELS = {
    "bad_boundary_mixing_group",
    "bad_edge_blurring_group",
    "bad_ranking_conflict_group",
    "opacity_regularization_candidate",
}
DRYRUN_ACTIONS = {"prune_candidate", "shrink_candidate", "split_candidate"}


def _read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@dataclass
class GaussianControlLoss:
    loss: torch.Tensor
    scalar: float
    count: int
    opacity_mean: float = 0.0
    opacity_min: float = 0.0
    opacity_max: float = 0.0


class GaussianControlManager:
    def __init__(self, cfg_node):
        self.cfg = cfg_node
        self.enabled = bool(getattr(cfg_node, "enabled", False))
        self.control_mode = str(getattr(cfg_node, "control_mode", "off"))
        self.evidence_source = str(getattr(cfg_node, "evidence_source", "auto"))
        self.risk_source = str(getattr(cfg_node, "risk_source", "da3_boundary"))
        self.supervision_mode = str(getattr(cfg_node, "supervision_mode", "da3_unsupervised"))
        self.counterfactual_objective = str(getattr(cfg_node, "counterfactual_objective", "da3_structure"))
        self.group_source = str(getattr(cfg_node, "group_source", "feedback_controller"))
        self.allow_parameter_modification = bool(getattr(cfg_node, "allow_parameter_modification", False))
        self.allow_real_prune = bool(getattr(cfg_node, "allow_real_prune", False))
        self.allow_real_split = bool(getattr(cfg_node, "allow_real_split", False))
        self.allow_real_shrink = bool(getattr(cfg_node, "allow_real_shrink", False))
        self.opacity_reg_weight = float(getattr(cfg_node, "opacity_reg_weight", 0.0))
        self.opacity_decay_factor = float(getattr(cfg_node, "opacity_decay_factor", 0.95))
        self.max_decay_gaussians_per_trigger = int(getattr(cfg_node, "max_decay_gaussians_per_trigger", 10))
        self.max_decay_ratio = float(getattr(cfg_node, "max_decay_ratio", 0.00005))
        self.max_controlled_gaussians = int(getattr(cfg_node, "max_controlled_gaussians", 256))
        self.min_group_confidence = float(getattr(cfg_node, "min_group_confidence", 0.0))
        self.min_multiview_support = int(getattr(cfg_node, "min_multiview_support", 1))
        self.min_support_pixels = int(getattr(cfg_node, "min_support_pixels", 8))
        self.rgb_safety_threshold = float(getattr(cfg_node, "rgb_safety_threshold", 0.02))
        self.protect_good_groups = bool(getattr(cfg_node, "protect_good_groups", True))
        self.skip_low_evidence = bool(getattr(cfg_node, "skip_low_evidence", True))
        self.group_evidence = []
        self.protected_ids = set()
        self.opacity_regularized = {}
        self.opacity_decay_candidates = {}
        self.dryrun_candidates = []
        self.last_decay_result = {"modified_count": 0, "modified_stable_ids": []}
        self._validate()

    def _validate(self):
        valid_modes = {
            "off",
            "protect_only",
            "opacity_regularization",
            "opacity_decay_apply",
            "repair_dryrun",
            "repair_apply",
            "prune_apply",
            "shrink_apply",
            "split_apply",
        }
        if self.control_mode not in valid_modes:
            raise ValueError(f"unknown gaussian_control.control_mode={self.control_mode!r}")
        if not self.enabled:
            return
        if self.control_mode in {"repair_apply", "prune_apply", "shrink_apply", "split_apply"}:
            raise RuntimeError("Real prune/shrink/split is not enabled in this stage.")
        if self.allow_real_prune or self.allow_real_split or self.allow_real_shrink:
            raise RuntimeError("Real prune/shrink/split is not enabled in this stage.")
        if self.control_mode == "opacity_decay_apply":
            if not self.allow_parameter_modification:
                raise RuntimeError("opacity_decay_apply requires train.gaussian_control.allow_parameter_modification=True")
            if not (0.0 < self.opacity_decay_factor < 1.0):
                raise ValueError("opacity_decay_factor must be in (0, 1)")
        elif self.allow_parameter_modification:
            raise RuntimeError("allow_parameter_modification=True is only allowed for opacity_decay_apply in this stage.")

    def normalize_group_record(self, record):
        label = str(record.get("group_label", "neutral_group"))
        action = str(record.get("future_action_tag", record.get("future_action", "skip")))
        support = int(record.get("support_pixels", record.get("support_pixel_count", 0)) or 0)
        confidence = float(record.get("confidence", 0.0) or 0.0)
        if confidence <= 0:
            confidence = min(1.0, float(record.get("group_risk_weighted_contribution", 0.0) or 0.0) / 10.0)
        rgb_delta = record.get("rgb_patch_mae_delta_mean", record.get("rgb_delta", None))
        rgb_delta = float(rgb_delta) if rgb_delta is not None else 0.0
        return {
            "group_id": str(record.get("group_id", record.get("region_key", "group"))),
            "stable_gaussian_ids": [int(v) for v in record.get("stable_gaussian_ids", [])],
            "view_ids": [str(record.get("view_id"))] if record.get("view_id") is not None else list(record.get("view_ids", [])),
            "region_ids": [str(record.get("region_id"))] if record.get("region_id") is not None else list(record.get("region_ids", [])),
            "risk_source": str(record.get("risk_source", self.risk_source)),
            "supervision_mode": str(record.get("supervision_mode", self.supervision_mode)),
            "counterfactual_objective": str(record.get("counterfactual_objective", self.counterfactual_objective)),
            "group_label": label,
            "future_action": action,
            "confidence": confidence,
            "risk_weighted_contribution": float(record.get("group_risk_weighted_contribution", 0.0) or 0.0),
            "mean_talpha": float(record.get("group_mean_talpha", 0.0) or 0.0),
            "max_talpha": float(record.get("group_max_talpha", 0.0) or 0.0),
            "support_pixels": support,
            "multiview_support": int(record.get("multiview_support", 1) or 1),
            "rgb_safety_score": float(max(0.0, self.rgb_safety_threshold - max(0.0, rgb_delta))),
            "rgb_delta": rgb_delta,
            "lidar_error_delta": record.get("lidar_error_delta"),
            "da3_structure_delta": record.get("da3_structure_delta_mean", record.get("da3_structure_delta")),
            "edge_delta": record.get("edge_delta"),
            "ranking_delta": record.get("ranking_delta"),
            "side_delta": record.get("side_delta"),
            "is_protected": label in PROTECT_LABELS or action in PROTECT_LABELS,
            "is_low_evidence": label == "low_evidence_group" or support < self.min_support_pixels,
            "metadata": record,
        }

    def update_from_group_evidence(self, group_records):
        self.group_evidence = [self.normalize_group_record(r) for r in group_records]
        self.protected_ids = self.build_protect_set()
        self.opacity_regularized = self.build_opacity_regularization_set()
        self.opacity_decay_candidates = self.build_opacity_decay_candidate_set()
        self.dryrun_candidates = self.build_dryrun_candidates()
        return self.get_control_summary()

    def update_from_group_evidence_file(self, path):
        if not path or not os.path.exists(path):
            return self.get_control_summary()
        return self.update_from_group_evidence(_read_json(path))

    def build_protect_set(self):
        protected = set()
        if not self.enabled or self.control_mode == "off" or not self.protect_good_groups:
            return protected
        for ev in self.group_evidence:
            if ev["is_protected"]:
                protected.update(ev["stable_gaussian_ids"])
        return protected

    def _eligible_bad(self, ev):
        if ev["is_low_evidence"] and self.skip_low_evidence:
            return False
        if ev["confidence"] < self.min_group_confidence:
            return False
        if ev["multiview_support"] < self.min_multiview_support:
            return False
        if ev["rgb_delta"] > self.rgb_safety_threshold:
            return False
        if ev["group_label"] not in BAD_LABELS and ev["future_action"] not in BAD_LABELS:
            return False
        return True

    def build_opacity_regularization_set(self):
        controlled = {}
        if not self.enabled or self.control_mode != "opacity_regularization":
            return controlled
        for ev in self.group_evidence:
            if not self._eligible_bad(ev):
                continue
            for gid in ev["stable_gaussian_ids"]:
                if gid in self.protected_ids:
                    continue
                controlled[gid] = max(float(ev["confidence"]), float(controlled.get(gid, 0.0)))
                if len(controlled) >= self.max_controlled_gaussians:
                    return controlled
        return controlled

    def build_opacity_decay_candidate_set(self):
        controlled = {}
        if not self.enabled or self.control_mode != "opacity_decay_apply":
            return controlled
        for ev in self.group_evidence:
            if not self._eligible_bad(ev):
                continue
            for gid in ev["stable_gaussian_ids"]:
                if gid in self.protected_ids:
                    continue
                prev = controlled.get(gid)
                if prev is None or ev["confidence"] > prev["confidence"]:
                    controlled[gid] = ev
                if len(controlled) >= self.max_controlled_gaussians:
                    return controlled
        return controlled

    def build_dryrun_candidates(self):
        rows = []
        if not self.enabled or self.control_mode not in {"repair_dryrun", "opacity_regularization", "opacity_decay_apply", "protect_only"}:
            return rows
        for ev in self.group_evidence:
            if ev["future_action"] in DRYRUN_ACTIONS or ev["group_label"] in BAD_LABELS:
                rows.append(ev)
        return rows[: self.max_controlled_gaussians]

    def compute_opacity_regularization_loss(self, model):
        zero = model.get_opacity.sum() * 0.0
        if not self.enabled or self.control_mode != "opacity_regularization" or self.opacity_reg_weight <= 0:
            return GaussianControlLoss(zero, 0.0, 0)
        if not self.opacity_regularized:
            return GaussianControlLoss(zero, 0.0, 0)
        stable_ids, _, _ = build_stable_id_map(model)
        row_ids = []
        weights = []
        for row, gid in enumerate(stable_ids):
            conf = self.opacity_regularized.get(int(gid))
            if conf is not None:
                row_ids.append(row)
                weights.append(float(conf))
        if not row_ids:
            return GaussianControlLoss(zero, 0.0, 0)
        device = model.get_opacity.device
        rows = torch.as_tensor(row_ids, device=device, dtype=torch.long)
        conf = torch.as_tensor(weights, device=device, dtype=model.get_opacity.dtype).view(-1, 1)
        opacity = model.get_opacity[rows]
        raw = torch.mean(opacity * conf)
        loss = self.opacity_reg_weight * raw
        return GaussianControlLoss(
            loss,
            float(loss.detach().item()),
            int(len(row_ids)),
            opacity_mean=float(opacity.detach().mean().item()),
            opacity_min=float(opacity.detach().min().item()),
            opacity_max=float(opacity.detach().max().item()),
        )

    def _stable_to_model_lookup(self, model):
        stable_ids, model_names, model_local = build_stable_id_map(model)
        lookup = {}
        for row, gid in enumerate(stable_ids):
            lookup[int(gid)] = {
                "row_id": int(row),
                "model_name": str(model_names[row]),
                "local_id": int(model_local[row]),
            }
        return lookup, stable_ids

    def apply_opacity_decay(self, model, output_dir, iteration=None):
        os.makedirs(output_dir, exist_ok=True)
        result = {
            "iteration": int(iteration) if iteration is not None else None,
            "control_mode": self.control_mode,
            "risk_source": self.risk_source,
            "supervision_mode": self.supervision_mode,
            "decay_factor": self.opacity_decay_factor,
            "max_decay_gaussians_per_trigger": self.max_decay_gaussians_per_trigger,
            "max_decay_ratio": self.max_decay_ratio,
            "selected_count": 0,
            "modified_count": 0,
            "skipped_count": 0,
            "protected_count": len(self.protected_ids),
            "gaussian_parameters_modified": False,
            "real_prune_enabled": False,
            "real_split_enabled": False,
            "real_shrink_enabled": False,
            "status": "skipped",
        }
        modified_rows = []
        skipped_rows = []
        protected_rows = [{"stable_gaussian_id": int(gid)} for gid in sorted(self.protected_ids)]
        if not self.enabled or self.control_mode != "opacity_decay_apply":
            result["reason"] = "opacity decay is disabled"
            self._write_decay_outputs(output_dir, result, modified_rows, protected_rows, skipped_rows)
            self.last_decay_result = {"modified_count": 0, "modified_stable_ids": []}
            return result
        if not self.allow_parameter_modification:
            raise RuntimeError("opacity_decay_apply refused because allow_parameter_modification=False")
        if self.allow_real_prune or self.allow_real_split or self.allow_real_shrink:
            raise RuntimeError("Real prune/shrink/split is not enabled in this stage.")
        if not self.opacity_decay_candidates:
            result["reason"] = "no eligible opacity decay candidates"
            self._write_decay_outputs(output_dir, result, modified_rows, protected_rows, skipped_rows)
            self.last_decay_result = {"modified_count": 0, "modified_stable_ids": []}
            return result

        lookup, stable_ids = self._stable_to_model_lookup(model)
        total_gaussians = int(len(stable_ids))
        ratio_cap = max(1, int(total_gaussians * self.max_decay_ratio))
        max_apply = max(0, min(self.max_decay_gaussians_per_trigger, ratio_cap, len(self.opacity_decay_candidates)))
        result["total_gaussians"] = total_gaussians
        result["ratio_cap"] = ratio_cap
        if max_apply <= 0:
            result["reason"] = "max decay cap is zero"
            self._write_decay_outputs(output_dir, result, modified_rows, protected_rows, skipped_rows)
            self.last_decay_result = {"modified_count": 0, "modified_stable_ids": []}
            return result

        ordered = sorted(
            self.opacity_decay_candidates.items(),
            key=lambda kv: (float(kv[1].get("confidence", 0.0)), float(kv[1].get("risk_weighted_contribution", 0.0))),
            reverse=True,
        )
        applied = 0
        modified_ids = []
        with torch.no_grad():
            for gid, ev in ordered:
                if applied >= max_apply:
                    skipped_rows.append(self._skip_row(gid, ev, "max_decay_cap_reached"))
                    continue
                if gid in self.protected_ids:
                    skipped_rows.append(self._skip_row(gid, ev, "protected"))
                    continue
                meta = lookup.get(int(gid))
                if meta is None:
                    skipped_rows.append(self._skip_row(gid, ev, "stable_id_not_in_current_graph"))
                    continue
                model_name = meta["model_name"]
                local_id = meta["local_id"]
                submodel = getattr(model, model_name, None)
                if submodel is None or not hasattr(submodel, "_opacity"):
                    skipped_rows.append(self._skip_row(gid, ev, "submodel_missing_opacity"))
                    continue
                if local_id < 0 or local_id >= int(submodel._opacity.shape[0]):
                    skipped_rows.append(self._skip_row(gid, ev, "local_id_out_of_range"))
                    continue
                old_opacity = submodel.get_opacity[local_id:local_id + 1].detach().clone()
                new_opacity = torch.clamp(old_opacity * self.opacity_decay_factor, min=1e-6, max=1.0 - 1e-6)
                new_raw = inverse_sigmoid(new_opacity)
                submodel._opacity.data[local_id:local_id + 1].copy_(new_raw)
                new_opacity_check = submodel.get_opacity[local_id:local_id + 1].detach().clone()
                modified_rows.append({
                    "iteration": result["iteration"],
                    "group_id": ev["group_id"],
                    "stable_gaussian_id": int(gid),
                    "view_local_id": int(meta["row_id"]),
                    "model_name": model_name,
                    "model_local_id": int(local_id),
                    "old_opacity": float(old_opacity.item()),
                    "new_opacity": float(new_opacity_check.item()),
                    "opacity_delta": float((new_opacity_check - old_opacity).item()),
                    "decay_factor": self.opacity_decay_factor,
                    "group_label": ev["group_label"],
                    "future_action": ev["future_action"],
                    "confidence": float(ev["confidence"]),
                    "support_pixels": int(ev["support_pixels"]),
                    "risk_source": ev["risk_source"],
                    "supervision_mode": ev["supervision_mode"],
                    "rgb_safety_score": float(ev["rgb_safety_score"]),
                    "is_protected": False,
                    "is_low_evidence": bool(ev["is_low_evidence"]),
                    "modified": True,
                })
                applied += 1
                modified_ids.append(int(gid))

        result["selected_count"] = len(self.opacity_decay_candidates)
        result["modified_count"] = len(modified_rows)
        result["skipped_count"] = len(skipped_rows)
        result["gaussian_parameters_modified"] = bool(modified_rows)
        result["status"] = "valid" if modified_rows else "skipped"
        self._write_decay_outputs(output_dir, result, modified_rows, protected_rows, skipped_rows)
        self.last_decay_result = {"modified_count": len(modified_rows), "modified_stable_ids": modified_ids}
        return result

    def _skip_row(self, gid, ev, reason):
        return {
            "stable_gaussian_id": int(gid),
            "group_id": ev.get("group_id", ""),
            "group_label": ev.get("group_label", ""),
            "future_action": ev.get("future_action", ""),
            "confidence": ev.get("confidence", 0.0),
            "support_pixels": ev.get("support_pixels", 0),
            "is_protected": int(gid) in self.protected_ids,
            "is_low_evidence": ev.get("is_low_evidence", False),
            "skip_reason": reason,
            "modified": False,
        }

    def _write_csv(self, path, rows):
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not rows:
            rows = []
        fields = sorted({k for row in rows for k in row.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_decay_outputs(self, output_dir, result, modified_rows, protected_rows, skipped_rows):
        _write_json(os.path.join(output_dir, "opacity_decay_apply_manifest.json"), result)
        self._write_csv(os.path.join(output_dir, "opacity_decay_gaussians.csv"), modified_rows)
        self._write_csv(os.path.join(output_dir, "protected_gaussians.csv"), protected_rows)
        self._write_csv(os.path.join(output_dir, "skipped_gaussians.csv"), skipped_rows)
        _write_json(os.path.join(output_dir, "gaussian_repair_audit.json"), {
            "status": result["status"],
            "gaussian_parameters_modified": result["gaussian_parameters_modified"],
            "real_prune_enabled": False,
            "real_split_enabled": False,
            "real_shrink_enabled": False,
            "modified_count": result["modified_count"],
            "skipped_count": result["skipped_count"],
            "protected_count": result["protected_count"],
            "safety_checks": {
                "only_opacity_decay_apply_can_modify": self.control_mode == "opacity_decay_apply",
                "allow_parameter_modification_explicit": self.allow_parameter_modification,
                "real_prune_disabled": not self.allow_real_prune,
                "real_split_disabled": not self.allow_real_split,
                "real_shrink_disabled": not self.allow_real_shrink,
            },
        })

    def get_control_summary(self):
        return {
            "enabled": self.enabled,
            "control_mode": self.control_mode,
            "risk_source": self.risk_source,
            "supervision_mode": self.supervision_mode,
            "counterfactual_objective": self.counterfactual_objective,
            "group_evidence_count": len(self.group_evidence),
            "protected_gaussian_count": len(self.protected_ids),
            "opacity_regularized_gaussian_count": len(self.opacity_regularized),
            "opacity_decay_candidate_count": len(self.opacity_decay_candidates),
            "last_opacity_decay_modified_count": int(self.last_decay_result.get("modified_count", 0)),
            "dryrun_repair_candidate_count": len(self.dryrun_candidates),
            "gaussian_parameters_modified": bool(self.last_decay_result.get("modified_count", 0)),
            "real_prune_enabled": False,
            "allow_parameter_modification": self.allow_parameter_modification,
            "allow_real_prune": self.allow_real_prune,
            "allow_real_split": self.allow_real_split,
            "allow_real_shrink": self.allow_real_shrink,
            "opacity_decay_factor": self.opacity_decay_factor,
        }

    def write_manifest(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        summary = self.get_control_summary()
        _write_json(os.path.join(output_dir, "gaussian_control_summary.json"), summary)
        _write_json(os.path.join(output_dir, "gaussian_control_group_evidence.json"), self.group_evidence)
        _write_json(os.path.join(output_dir, "gaussian_control_dryrun_candidates.json"), self.dryrun_candidates)
        _write_json(os.path.join(output_dir, "gaussian_control_opacity_decay_candidates.json"), list(self.opacity_decay_candidates.values()))
        return summary

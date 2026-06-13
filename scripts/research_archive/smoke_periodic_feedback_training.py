import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def main():
    parser = argparse.ArgumentParser(description="Run a tiny training-loop smoke for periodic feedback controller.")
    parser.add_argument("--a5000-dir", default="output/local_formal/p15_allcam_A_da3_only_5000")
    parser.add_argument("--output-dir", default="output/local_feedback/periodic_feedback_training_smoke")
    parser.add_argument("--iterations", type=int, default=5002)
    parser.add_argument("--trigger-iter", type=int, default=5001)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--max-triggers", type=int, default=1)
    parser.add_argument("--max-regions", type=int, default=1)
    parser.add_argument("--top-contributors", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--run-counterfactual", action="store_true")
    parser.add_argument("--dynamic-recompute", action="store_true")
    parser.add_argument("--risk-source", default="da3_boundary", choices=["da3_boundary", "da3_structure", "lidar_error"])
    parser.add_argument("--supervision-mode", default="da3_unsupervised", choices=["da3_unsupervised", "lidar_supervised", "hybrid_reference"])
    parser.add_argument("--gaussian-control-mode", default="off", choices=["off", "protect_only", "opacity_regularization", "opacity_decay_apply", "repair_dryrun"])
    parser.add_argument("--gaussian-control-opacity-weight", type=float, default=0.0)
    parser.add_argument("--gaussian-control-opacity-decay-factor", type=float, default=0.95)
    parser.add_argument("--gaussian-control-max-decay", type=int, default=10)
    parser.add_argument("--gaussian-control-max-decay-ratio", type=float, default=0.00005)
    parser.add_argument("--gaussian-control-allow-parameter-modification", action="store_true")
    parser.add_argument("--gaussian-control-counterfactual-objective", default="", choices=["", "da3_structure", "lidar_depth_error", "hybrid"])
    parser.add_argument("--scalar-trace-path", default="")
    args = parser.parse_args()

    a5000 = resolve(args.a5000_dir)
    out = resolve(args.output_dir)
    cfg_path = a5000 / "configs" / "config_000000.yaml"
    src_ckpt = a5000 / "trained_model" / "iteration_5000.pth"
    dst_ckpt_dir = out / "trained_model"
    dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_ckpt, dst_ckpt_dir / "iteration_5000.pth")

    signal_path = "output/local_feedback/da3_boundary_soft_contribution_feedback_A5000_top30/da3_contribution_softpatch_feedback_signal.json"
    contribution_path = "output/local_feedback/da3_boundary_contribution_debug_A5000_top30_regionpixels/contribution_responsibility_all_views_summary.json"
    dryrun_script = "script/dryrun_da3_structure_counterfactual_scorer.py"

    cmd = [
        sys.executable,
        "train.py",
        "--config",
        str(cfg_path),
        "mode", "train",
        "model_path", str(out),
        "loaded_iter", "5000",
        "train.iterations", str(args.iterations),
        "train.scalar_trace_path", args.scalar_trace_path,
        "train.disable_structure_updates", "True",
        "train.guided_feedback.enabled", "True",
        "train.guided_feedback.signal_path", signal_path,
        "train.guided_feedback.supervision_mode", args.supervision_mode,
        "train.guided_feedback.use_lidar_depth", "True" if args.supervision_mode in {"lidar_supervised", "hybrid_reference"} else "False",
        "train.guided_feedback.use_da3_structure", "True" if args.supervision_mode in {"da3_unsupervised", "hybrid_reference"} else "False",
        "train.guided_feedback.assert_no_lidar_supervision", "True" if args.supervision_mode == "da3_unsupervised" else "False",
        "train.guided_feedback.feedback_mode", "contribution_specific",
        "optim.lambda_depth_lidar", "0.0",
        "geovit.enabled", "True",
        "geovit.model_dir", "../.cache/huggingface/hub/models--depth-anything--DA3-LARGE-1.1/snapshots/main",
        "geovit.local_files_only", "True",
        "train.feedback_controller.enabled", "True",
        "train.feedback_controller.mode", "repair_dryrun" if args.run_counterfactual else "feedback_update",
        "train.feedback_controller.start_iter", str(args.trigger_iter),
        "train.feedback_controller.interval", str(args.interval),
        "train.feedback_controller.max_triggers", str(args.max_triggers),
        "train.feedback_controller.risk_source", args.risk_source,
        "train.feedback_controller.supervision_mode", args.supervision_mode,
        "train.feedback_controller.feedback_mode", "contribution_softpatch",
        "train.feedback_controller.repair_mode", "dryrun" if args.run_counterfactual else "none",
        "train.feedback_controller.allow_parameter_modification", "False",
        "train.feedback_controller.output_dir", str(out / "feedback_controller"),
        "train.feedback_controller.max_regions", str(args.max_regions),
        "train.feedback_controller.top_contributors", str(args.top_contributors),
        "train.feedback_controller.signal_path", signal_path,
        "train.feedback_controller.contribution_summary_path", contribution_path,
        "train.feedback_controller.dryrun_scorer_path", dryrun_script,
        "train.feedback_controller.run_counterfactual", "True" if args.run_counterfactual else "False",
        "train.feedback_controller.run_dryrun_scorer", "True" if args.run_counterfactual else "False",
        "train.feedback_controller.run_candidate_tagging", "True" if args.run_counterfactual else "False",
        "train.feedback_controller.fail_policy", "stop",
    ]
    if args.gaussian_control_mode != "off":
        objective = args.gaussian_control_counterfactual_objective
        if not objective:
            objective = "lidar_depth_error" if args.risk_source == "lidar_error" else "da3_structure"
        cmd += [
            "train.gaussian_control.enabled", "True",
            "train.gaussian_control.control_mode", args.gaussian_control_mode,
            "train.gaussian_control.evidence_source", "lidar" if args.risk_source == "lidar_error" else "da3",
            "train.gaussian_control.risk_source", args.risk_source,
            "train.gaussian_control.supervision_mode", args.supervision_mode,
            "train.gaussian_control.counterfactual_objective", objective,
            "train.gaussian_control.group_source", "feedback_controller",
            "train.gaussian_control.allow_parameter_modification", "True" if args.gaussian_control_allow_parameter_modification else "False",
            "train.gaussian_control.allow_real_prune", "False",
            "train.gaussian_control.allow_real_split", "False",
            "train.gaussian_control.allow_real_shrink", "False",
            "train.gaussian_control.opacity_reg_weight", str(args.gaussian_control_opacity_weight),
            "train.gaussian_control.opacity_decay_factor", str(args.gaussian_control_opacity_decay_factor),
            "train.gaussian_control.max_decay_gaussians_per_trigger", str(args.gaussian_control_max_decay),
            "train.gaussian_control.max_decay_ratio", str(args.gaussian_control_max_decay_ratio),
        ]
    if args.dynamic_recompute:
        cmd += [
            "train.feedback_controller.feedback_mode", "group_softpatch",
            "train.feedback_controller.contribution_source", "live_current_model",
            "train.feedback_controller.recompute_risk", "True",
            "train.feedback_controller.recompute_contribution", "True",
            "train.feedback_controller.recompute_responsible_groups", "True",
            "train.feedback_controller.recompute_softpatch", "True",
        ]
    if args.run_counterfactual:
        cmd += [
            "train.feedback_controller.dryrun_extra_args",
            "--top-contributors,{0},--max-candidates,{1},--config,{2},mode,evaluate,model_path,{3},loaded_iter,5000,geovit.enabled,True,geovit.model_dir,../.cache/huggingface/hub/models--depth-anything--DA3-LARGE-1.1/snapshots/main,geovit.local_files_only,True".format(
            args.top_contributors,
            args.max_candidates,
            cfg_path,
            a5000,
            ),
        ]
    env = os.environ.copy()
    env["PWD"] = str(Path.cwd())
    subprocess.run(cmd, cwd=str(Path.cwd()), env=env, check=True)
    print(f"smoke output: {out}")


if __name__ == "__main__":
    main()

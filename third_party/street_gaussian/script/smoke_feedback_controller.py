import argparse
import json
import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PWD", PROJECT_ROOT)

from lib.utils.feedback_controller import make_periodic_feedback_controller  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/local_feedback/feedback_controller_smoke")
    parser.add_argument("--iteration", type=int, default=5001)
    parser.add_argument("--mode", default="diagnose_only", choices=["diagnose_only", "feedback_update", "repair_dryrun"])
    parser.add_argument("--risk-source", default="da3_boundary", choices=["lidar_error", "da3_boundary", "da3_structure"])
    parser.add_argument("--supervision-mode", default="da3_unsupervised", choices=["lidar_supervised", "da3_unsupervised", "hybrid_reference"])
    parser.add_argument("--signal-path", default="output/local_feedback/da3_boundary_soft_contribution_feedback_A5000_top30/da3_contribution_softpatch_feedback_signal.json")
    parser.add_argument("--contribution-summary-path", default="output/local_feedback/da3_boundary_contribution_debug_A5000_top30_regionpixels/contribution_responsibility_all_views_summary.json")
    parser.add_argument("--dryrun-scorer-path", default="script/dryrun_da3_structure_counterfactual_scorer.py")
    parser.add_argument("--run-dryrun-scorer", action="store_true")
    parser.add_argument("--top-regions", type=int, default=1)
    parser.add_argument("--top-contributors", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--a5000-config", default="output/local_formal/p15_allcam_A_da3_only_5000/configs/config_000000.yaml")
    parser.add_argument("--a5000-model-path", default="output/local_formal/p15_allcam_A_da3_only_5000")
    parser.add_argument("--da3-model-dir", default="../.cache/huggingface/hub/models--depth-anything--DA3-LARGE-1.1/snapshots/main")
    args = parser.parse_args()

    dryrun_extra = [
        "--top-contributors", str(args.top_contributors),
        "--max-candidates", str(args.max_candidates),
        "--config", args.a5000_config,
        "mode", "evaluate",
        "model_path", args.a5000_model_path,
        "loaded_iter", "5000",
        "geovit.enabled", "True",
        "geovit.model_dir", args.da3_model_dir,
        "geovit.local_files_only", "True",
    ]
    cfg_node = SimpleNamespace(
        enabled=True,
        start_iter=args.iteration,
        interval=1,
        mode=args.mode,
        risk_source=args.risk_source,
        supervision_mode=args.supervision_mode,
        repair_mode="dryrun" if args.mode == "repair_dryrun" else "none",
        output_dir=args.output_dir,
        selected_views=[],
        max_regions=args.top_regions,
        signal_path=args.signal_path,
        contribution_summary_path=args.contribution_summary_path,
        dryrun_scorer_path=args.dryrun_scorer_path,
        dryrun_extra_args=dryrun_extra,
        run_dryrun_scorer=args.run_dryrun_scorer,
        allow_parameter_modification=False,
        skip_existing=False,
    )
    controller = make_periodic_feedback_controller(cfg_node, model_path=args.output_dir)
    result = controller.trigger(
        args.iteration,
        selected_views=["000000_1"],
        selected_regions=["smoke_da3_boundary_region"],
        extra={
            "smoke_test": True,
            "expected_gaussian_parameters_modified": False,
            "note": "This validates manifest/dry-run scheduling only; no training or repair is executed.",
        },
    )
    print(json.dumps({"status": result.status, "manifest_path": result.manifest_path}, indent=2))


if __name__ == "__main__":
    main()

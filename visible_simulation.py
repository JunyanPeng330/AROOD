"""Standalone visible evaluation launcher for AROOD.

This script delegates simulation, checkpoint loading, video capture, and JSONL
logging to A_Video.py. It does not train or update the policy.
Run from the repository root so relative paths such as policy/A_1.pt resolve.
"""

import argparse
import os

import A_Video


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one visible AROOD evaluation episode and record an MP4 "
            "without training the policy."
        )
    )
    parser.add_argument(
        "--model-path",
        default=A_Video.MODEL_PATH,
        help=f"Trained checkpoint path (default: {A_Video.MODEL_PATH}).",
    )
    parser.add_argument(
        "--video-dir",
        default=os.path.join("figures", "videos"),
        help="Directory where the MP4 will be saved.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=1,
        help="1-based output/rollout label; does not select a historical training episode.",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=133,
        help="Environment seed for repeatable scenario initialization (default: 133).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=A_Video.SEED,
        help="Base seed used if --scenario-seed is explicitly set to an empty value in code.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=A_Video.MAX_STEPS,
        help=f"Maximum evaluation steps (default: {A_Video.MAX_STEPS}).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using deterministic choices.",
    )
    parser.add_argument(
        "--no-store-simulator",
        action="store_true",
        help="Disable persistent JSONL simulator logging.",
    )
    parser.add_argument(
        "--simulator-record-path",
        default=A_Video.SIMULATOR_RECORD_PATH,
        help=(
            "Persistent append-only JSONL path "
            f"(default: {A_Video.SIMULATOR_RECORD_PATH})."
        ),
    )
    args = parser.parse_args()

    print("[VISIBLE SIMULATION] evaluation-only; policy training is not invoked.")
    print(f"[VISIBLE SIMULATION] checkpoint: {os.path.abspath(args.model_path)}")
    print(f"[VISIBLE SIMULATION] video directory: {os.path.abspath(args.video_dir)}")
    print(f"[VISIBLE SIMULATION] scenario seed: {args.scenario_seed}")
    print(
        "[VISIBLE SIMULATION] simulator log: "
        + ("disabled" if args.no_store_simulator else os.path.abspath(args.simulator_record_path))
    )

    result = A_Video.evaluate_and_record_video(
        model_path=args.model_path,
        video_dir=args.video_dir,
        seed=args.seed,
        episode_index=args.episode_index,
        scenario_seed=args.scenario_seed,
        deterministic=not args.stochastic,
        max_steps=args.max_steps,
        store_simulator=not args.no_store_simulator,
        simulator_record_path=args.simulator_record_path,
    )

    print("[VISIBLE SIMULATION COMPLETE]")
    print(f"  steps: {result['steps']} / {result['max_steps_requested']}")
    print(f"  collision: {result['collision']}")
    print(f"  video directory: {result['video_dir']}")
    if result.get("simulator_record_path"):
        print(f"  simulator log: {result['simulator_record_path']}")


if __name__ == "__main__":
    main()

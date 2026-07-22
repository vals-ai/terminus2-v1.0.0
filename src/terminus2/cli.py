"""CLI entry point for the Terminus 2 agent."""

import argparse
import asyncio
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="terminus2",
        description="Terminus 2 - Terminal-based AI agent",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run the agent on a task")
    run_parser.add_argument(
        "instruction",
        type=str,
        nargs="?",
        default=None,
        help="The task instruction for the agent",
    )
    run_parser.add_argument(
        "--problem-path",
        type=Path,
        default=None,
        help="Path to a file containing the task instruction",
    )
    run_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name to use (e.g. 'anthropic/claude-sonnet-4-20250514')",
    )
    run_parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to store logs and trajectories (default: ./logs)",
    )
    run_parser.add_argument(
        "--parser",
        type=str,
        choices=["json", "xml"],
        default="json",
        help="Response parser format (default: json)",
    )
    run_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of agent turns",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the model",
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum output tokens for the model",
    )
    run_parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable/disable reasoning (e.g. Anthropic extended thinking). If omitted, the model's default is used.",
    )
    run_parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high", "default", "xhigh"],
        default=None,
        help="Reasoning effort for models that support reasoning effort. If omitted, the model's default is used.",
    )
    run_parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL for the API endpoint",
    )
    run_parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Disable context summarization",
    )
    run_parser.add_argument(
        "--raw-trajectory",
        action="store_true",
        help="Save raw LLM responses in trajectory (for SFT data export)",
    )
    run_parser.add_argument(
        "--linear-history",
        action="store_true",
        help="Split trajectory files on context summarization",
    )

    # --- validate command ---
    validate_parser = subparsers.add_parser(
        "validate", help="Validate an ATIF trajectory file"
    )
    validate_parser.add_argument(
        "trajectory_file",
        type=str,
        help="Path to the trajectory JSON file to validate",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "validate":
        _run_validate(args)
    elif args.command == "run":
        if args.problem_path:
            args.instruction = args.problem_path.read_text()
        if not args.instruction:
            run_parser.error("either instruction or --problem-path is required")
        asyncio.run(_run_agent(args))


async def _run_agent(args):
    from terminus2.terminus_2 import Terminus2
    from terminus2.agent.context import AgentContext
    from terminus2.environment_local import LocalEnvironment
    from terminus2.model_patch import capture_model_patch_baseline, write_model_patch
    from terminus2.trial.paths import TrialPaths

    logs_dir = args.logs_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    trajectory_config = {}
    if args.raw_trajectory:
        trajectory_config["raw_content"] = True
    if args.linear_history:
        trajectory_config["linear_history"] = True

    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=args.model,
        parser_name=args.parser,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
        reasoning_effort=args.reasoning_effort,
        api_base=args.api_base,
        enable_summarize=not args.no_summarize,
        trajectory_config=trajectory_config or None,
    )

    trial_paths = TrialPaths(trial_dir=logs_dir)
    environment = LocalEnvironment(trial_paths=trial_paths)

    await environment.start(force_build=False)
    await agent.setup(environment)

    # cd the tmux session to the actual working directory (bash --login resets cwd)
    import os

    cwd = os.getcwd()
    repo = Path(cwd)
    private_paths = [logs_dir]
    if args.problem_path is not None:
        private_paths.append(args.problem_path.absolute())
    baseline = capture_model_patch_baseline(repo, excluded_paths=tuple(private_paths))
    await agent._session.send_keys(keys=[f"cd {cwd}", "Enter"])
    import asyncio as _asyncio

    await _asyncio.sleep(0.5)

    context = AgentContext()
    print(f"Terminus 2 agent initialized with model: {args.model}")
    print(f"Logs directory: {logs_dir}")

    await agent.run(args.instruction, environment, context)

    if baseline is not None:
        _ = write_model_patch(
            repo,
            logs_dir,
            logs_dir / "trajectory.json",
            baseline,
        )

    await environment.stop(delete=False)


def _run_validate(args):
    from terminus2.utils.trajectory_validator import TrajectoryValidator

    trajectory_path = Path(args.trajectory_file)

    if not trajectory_path.exists():
        print(f"Error: File not found: {trajectory_path}", file=sys.stderr)
        sys.exit(1)

    validator = TrajectoryValidator()

    try:
        is_valid = validator.validate(trajectory_path)

        if is_valid:
            print(f"Trajectory is valid: {trajectory_path}")
            sys.exit(0)
        else:
            print(
                f"Trajectory validation failed: {trajectory_path}", file=sys.stderr
            )
            print(f"\nFound {len(validator.errors)} error(s):", file=sys.stderr)
            for error in validator.errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

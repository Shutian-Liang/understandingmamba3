from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
PAPERCODE_ROOT = REPO_ROOT.parent
CONFIG_ROOT = REPO_ROOT / "configs"
TASK_SELECTORS = ("parity", "mod3", "arithmetic")
MODEL_SELECTORS = (
    "mamba2",
    "no_rotation",
    "static_rotation",
    "overdamped",
    "critical",
    "underdamped",
)

SECTION_KEYS = {
    "model": {
        "siso_backend",
        "n_layers",
        "d_intermediate",
        "tie_embeddings",
        "d_state",
        "expand",
        "headdim",
        "rope_fraction",
        "mimo_rank",
        "chunk_size",
        "transition_mode",
        "overdamped_rho_scale",
        "second_order_transition",
        "scan_backend",
        "scan_chunk_size",
    },
    "training": {
        "steps",
        "batch_size",
        "micro_batch_size",
        "curriculum_min_length",
        "curriculum_start_max",
        "curriculum_end_max",
        "curriculum_ramp_start_step",
        "curriculum_ramp_steps",
        "long_sequence_mix_ratio",
        "lr",
        "min_lr",
        "warmup_steps",
        "lr_decay_horizon_steps",
        "cooldown_steps",
        "scheduler",
        "weight_decay",
        "grad_clip",
    },
    "evaluation": {
        "context_length",
        "log_every",
        "eval_every",
        "save_every",
        "validation_samples",
        "test_samples",
        "eval_lengths",
    },
    "runtime": {
        "output_root",
        "precision",
        "allow_tf32",
        "compile_model",
        "compile_mode",
        "device",
        "resume",
    },
}
ROOT_KEYS = {
    "schema_version",
    "models",
    "experiment",
    "model",
    "training",
    "evaluation",
    "runtime",
    "sweep",
}


def csv_values(value: str, cast: type) -> list[Any]:
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def model_names(value: str) -> list[str]:
    models = csv_values(value, str)
    supported = {
        "mamba3_fixed_rope",
        "mamba3_no_rope",
        "mamba3_second_order",
        "mamba2",
    }
    unknown = sorted(set(models) - supported)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported model(s): {', '.join(unknown)}; choose from "
            "mamba3_fixed_rope,mamba3_no_rope,mamba3_second_order,mamba2"
        )
    return models


def resolve_config(value: str | Path) -> Path:
    requested = Path(value)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(REPO_ROOT / requested)
        candidates.append(CONFIG_ROOT / requested)
        if requested.suffix != ".json":
            candidates.append(CONFIG_ROOT / f"{requested}.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"config not found; searched: {searched}")


def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"config section {section!r} must be a JSON object")
    return value


def _validate_mimo_rank(value: Any) -> int:
    if type(value) is not int or value != 1:
        raise ValueError("model.mimo_rank must be 1 for the paper state-tracking experiments")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a JSON object")
    unknown_root = sorted(set(config) - ROOT_KEYS)
    if unknown_root:
        raise ValueError(f"unknown config section(s): {', '.join(unknown_root)}")
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty JSON array")
    unknown_models = sorted(
        set(models)
        - {
            "mamba3_fixed_rope",
            "mamba3_no_rope",
            "mamba3_second_order",
            "mamba2",
        }
    )
    if unknown_models:
        raise ValueError(
            "unsupported config model(s): " + ", ".join(unknown_models)
        )

    experiment = _require_mapping(config, "experiment")
    task = experiment.get("task")
    if task not in {"parity", "mod3_counting", "modular_arithmetic_mod3"}:
        raise ValueError(f"Unsupported state-tracking task: {task!r}")

    for section, allowed_keys in SECTION_KEYS.items():
        values = _require_mapping(config, section)
        unknown = sorted(set(values) - allowed_keys)
        if unknown:
            raise ValueError(
                f"unknown key(s) in {section}: {', '.join(unknown)}"
            )

    if config["training"].get("long_sequence_mix_ratio", 0.0) != 0.0:
        raise ValueError("The paper recipes require long_sequence_mix_ratio=0")
    model = config["model"]
    _validate_mimo_rank(model.get("mimo_rank"))
    rho_scale = model.get("overdamped_rho_scale", 1.0)
    if not isinstance(rho_scale, (int, float)) or not 0.0 < rho_scale <= 1.0:
        raise ValueError("model.overdamped_rho_scale must be in (0, 1]")

    sweep = _require_mapping(config, "sweep")
    for key in ("d_models", "learning_rates", "seeds"):
        values = sweep.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{key} must be a non-empty JSON array")
    return config


def _option_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _append_option(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    option = _option_name(key)
    if isinstance(value, bool):
        command.append(option if value else "--no-" + key.replace("_", "-"))
    elif isinstance(value, list):
        command.extend((option, ",".join(str(item) for item in value)))
    else:
        command.extend((option, str(value)))


def flattened_options(config: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for section in SECTION_KEYS:
        options.update(config[section])
    output_root = Path(options["output_root"])
    if not output_root.is_absolute():
        options["output_root"] = REPO_ROOT / output_root
    resume = options.get("resume")
    if resume not in (None, "auto", "none"):
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            options["resume"] = REPO_ROOT / resume_path
    return options


def build_train_command(
    config: dict[str, Any],
    *,
    model: str,
    d_model: int,
    learning_rate: float,
    seed: int,
    python_bin: str,
    overrides: dict[str, Any] | None = None,
    trainer_args: Iterable[str] = (),
) -> list[str]:
    if model not in {
        "mamba3_fixed_rope",
        "mamba3_no_rope",
        "mamba3_second_order",
        "mamba2",
    }:
        raise ValueError(f"unsupported model: {model}")
    options = flattened_options(config)
    options.update(
        {"d_model": d_model, "lr": learning_rate, "seed": seed}
    )
    if overrides:
        options.update({key: value for key, value in overrides.items() if value is not None})
    _validate_mimo_rank(options.get("mimo_rank"))

    command = [
        python_bin,
        "-m",
        "state_tracking.train",
        "--model",
        model,
        "--task",
        str(config["experiment"]["task"]),
    ]
    for key, value in options.items():
        _append_option(command, key, value)
    command.extend(trainer_args)
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        help=(
            "Config name (for example 'parity/underdamped') or path to a JSON "
            "config below configs/. Omit it when using --task/--model."
        ),
    )
    parser.add_argument(
        "--task", choices=("all",) + TASK_SELECTORS, default="all"
    )
    parser.add_argument(
        "--model", choices=("all",) + MODEL_SELECTORS, default="all"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--models",
        type=model_names,
        help=(
            "Comma-separated models: mamba3_fixed_rope,mamba3_no_rope,"
            "mamba3_second_order,mamba2. "
            "Defaults to the config's models array."
        ),
    )
    parser.add_argument("--d-models", type=lambda value: csv_values(value, int))
    parser.add_argument(
        "--learning-rates", type=lambda value: csv_values(value, float)
    )
    parser.add_argument("--seeds", type=lambda value: csv_values(value, int))
    parser.add_argument("--lr", type=float, help="Run one learning rate")
    parser.add_argument("--seed", type=int, help="Run one seed")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--curriculum-ramp-start-step", type=int)
    parser.add_argument("--curriculum-ramp-steps", type=int)
    parser.add_argument("--long-sequence-mix-ratio", type=float, choices=(0.0,))
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--d-intermediate", type=int)
    parser.add_argument(
        "--tie-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--d-state", type=int)
    parser.add_argument("--expand", type=int)
    parser.add_argument("--headdim", type=int)
    parser.add_argument("--rope-fraction", type=float)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument(
        "--transition-mode",
        choices=(
            "matched_underdamped",
            "critically_damped",
            "overdamped",
        ),
    )
    parser.add_argument("--overdamped-rho-scale", type=float)
    parser.add_argument(
        "--second-order-transition", choices=("matrix_exp", "closed_form")
    )
    parser.add_argument(
        "--scan-backend",
        choices=(
            "sequential",
            "parallel",
            "triton",
            "triton_associative",
            "triton_legacy",
        ),
    )
    parser.add_argument("--scan-chunk-size", type=int)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--lr-decay-horizon-steps", type=int)
    parser.add_argument("--cooldown-steps", type=int)
    parser.add_argument("--scheduler", choices=("cosine", "constant"))
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--eval-lengths", type=lambda value: csv_values(value, int))
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--siso-backend", choices=("triton", "torch_reference"))
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
    )
    parser.add_argument("--resume")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def selected_recipes(args: argparse.Namespace) -> list[tuple[Path, str | None, str | None]]:
    if args.config is not None:
        if args.task != "all" or args.model != "all":
            raise ValueError("use either a config path or --task/--model selectors")
        return [(resolve_config(args.config), None, None)]

    recipes = []
    for task in TASK_SELECTORS:
        if args.task not in ("all", task):
            continue
        for model in MODEL_SELECTORS:
            if args.model not in ("all", model):
                continue
            recipes.append((CONFIG_ROOT / task / f"{model}.json", task, model))
    return recipes


def main() -> int:
    raw_args = sys.argv[1:]
    if "--" in raw_args:
        separator = raw_args.index("--")
        launcher_args = raw_args[:separator]
        trainer_args = raw_args[separator + 1 :]
    else:
        launcher_args = raw_args
        trainer_args = []
    args = build_parser().parse_args(launcher_args)
    if args.lr is not None and args.learning_rates is not None:
        raise ValueError("use either --lr or --learning-rates")
    if args.seed is not None and args.seeds is not None:
        raise ValueError("use either --seed or --seeds")

    recipes = selected_recipes(args)
    if args.list:
        for config_path, task_selector, model_selector in recipes:
            config = load_config(config_path)
            sweep = config["sweep"]
            runs = (
                len(config["models"])
                * len(sweep["d_models"])
                * len(sweep["learning_rates"])
                * len(sweep["seeds"])
            )
            label = (
                f"{task_selector:10s} {model_selector:16s}"
                if task_selector is not None and model_selector is not None
                else config_path.stem
            )
            try:
                displayed_path = config_path.relative_to(REPO_ROOT)
            except ValueError:
                displayed_path = config_path
            print(f"{label} {runs:2d} runs  {displayed_path}")
        return 0

    protected = {"--model", "--task", "--mimo-rank", "--is-mimo"}
    forbidden = sorted(
        argument
        for argument in trainer_args
        if argument.split("=", 1)[0] in protected
    )
    if forbidden:
        raise ValueError(
            "launcher-controlled argument(s) cannot be overridden: "
            + ", ".join(forbidden)
        )

    common_overrides = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "curriculum_ramp_start_step": args.curriculum_ramp_start_step,
        "curriculum_ramp_steps": args.curriculum_ramp_steps,
        "long_sequence_mix_ratio": args.long_sequence_mix_ratio,
        "n_layers": args.n_layers,
        "d_intermediate": args.d_intermediate,
        "tie_embeddings": args.tie_embeddings,
        "d_state": args.d_state,
        "expand": args.expand,
        "headdim": args.headdim,
        "rope_fraction": args.rope_fraction,
        "chunk_size": args.chunk_size,
        "transition_mode": args.transition_mode,
        "overdamped_rho_scale": args.overdamped_rho_scale,
        "second_order_transition": args.second_order_transition,
        "scan_backend": args.scan_backend,
        "scan_chunk_size": args.scan_chunk_size,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "lr_decay_horizon_steps": args.lr_decay_horizon_steps,
        "cooldown_steps": args.cooldown_steps,
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "context_length": args.context_length,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "validation_samples": args.validation_samples,
        "test_samples": args.test_samples,
        "eval_lengths": args.eval_lengths,
        "precision": args.precision,
        "siso_backend": args.siso_backend,
        "allow_tf32": args.allow_tf32,
        "compile_model": args.compile_model,
        "compile_mode": args.compile_mode,
        "resume": args.resume,
    }
    exit_code = 0
    for config_path, task_selector, model_selector in recipes:
        config = load_config(config_path)
        sweep = config["sweep"]
        models = args.models or config["models"]
        d_models = args.d_models or sweep["d_models"]
        learning_rates = (
            [args.lr]
            if args.lr is not None
            else args.learning_rates or sweep["learning_rates"]
        )
        seeds = [args.seed] if args.seed is not None else args.seeds or sweep["seeds"]
        overrides = dict(common_overrides)
        if task_selector is not None and model_selector is not None:
            output_root = args.output_root or REPO_ROOT / "outputs"
            if not output_root.is_absolute():
                output_root = REPO_ROOT / output_root
            overrides["output_root"] = output_root / task_selector / model_selector
        elif args.output_root is not None:
            overrides["output_root"] = args.output_root

        commands = [
            build_train_command(
                config,
                model=model,
                d_model=d_model,
                learning_rate=learning_rate,
                seed=seed,
                python_bin=sys.executable,
                overrides=overrides,
                trainer_args=trainer_args,
            )
            for model, d_model, learning_rate, seed in itertools.product(
                models, d_models, learning_rates, seeds
            )
        ]
        print(f"Config: {config_path}")
        print(
            f"Task: {config['experiment']['task']}; models: {','.join(models)}; "
            f"runs: {len(commands)}"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env.setdefault(
            "TILELANG_CACHE_DIR", str(REPO_ROOT / ".cache" / "tilelang")
        )
        effective_options = flattened_options(config)
        if overrides.get("allow_tf32") is not None:
            effective_options["allow_tf32"] = overrides["allow_tf32"]
        if not effective_options.get("allow_tf32", True):
            env["TRITON_F32_DEFAULT"] = "ieee"
        else:
            env.pop("TRITON_F32_DEFAULT", None)

        for index, command in enumerate(commands, start=1):
            printable = (
                f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} "
                f"{shlex.join(command)}"
            )
            print(f"[{index}/{len(commands)}] {printable}", flush=True)
            if args.dry_run:
                continue
            completed = subprocess.run(
                command, cwd=PAPERCODE_ROOT, env=env, check=False
            )
            if completed.returncode:
                exit_code = completed.returncode
                if not args.continue_on_error:
                    return exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

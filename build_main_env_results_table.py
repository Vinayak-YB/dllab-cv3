import argparse
import csv
import json
import os
from pathlib import Path


RUNS = [
    # ---------------------------------------------------------
    # Baseline bouncing ball
    # ---------------------------------------------------------
    {
        "environment": "baseline",
        "model_type": "lstm",
        "run": "envv2_baseline_lstm",
        "path": (
            "models/envv2_baseline_lstm/"
            "physics/evaluation_results_test.json"
        ),
    },
    {
        "environment": "baseline",
        "model_type": "gru",
        "run": "envv2_baseline_gru",
        "path": (
            "models/envv2_baseline_gru/"
            "physics/evaluation_results_test.json"
        ),
    },
    {
        "environment": "baseline",
        "model_type": "latent_flow",
        "run": "envv2_baseline_latent_flow",
        "path": (
            "checkpoints/envv2_baseline_latent_flow/"
            "physics/evaluation_results_test.json"
        ),
    },

    # ---------------------------------------------------------
    # Magnetic wells
    # ---------------------------------------------------------
    {
        "environment": "magnetic_wells",
        "model_type": "lstm",
        "run": "envv2_magwells_lstm",
        "path": (
            "models/envv2_magwells_lstm/"
            "physics/evaluation_results_test.json"
        ),
    },
    {
        "environment": "magnetic_wells",
        "model_type": "gru",
        "run": "envv2_magwells_gru",
        "path": (
            "models/envv2_magwells_gru/"
            "physics/evaluation_results_test.json"
        ),
    },
    {
        "environment": "magnetic_wells",
        "model_type": "latent_flow",
        "run": "envv2_magwells_latent_flow",
        "path": (
            "checkpoints/envv2_magwells_latent_flow/"
            "physics/evaluation_results_test.json"
        ),
    },

    # ---------------------------------------------------------
    # Randomized billiard
    # ---------------------------------------------------------
    {
        "environment": "billiard",
        "model_type": "lstm",
        "run": "envv2_billiard_lstm",
        "path": (
            "models/envv2_billiard_lstm/"
            "physics/evaluation_results_test.json"
        ),
    },
    {
        "environment": "billiard",
        "model_type": "gru",
        "run": "envv2_billiard_gru",
        "path": (
            "models/envv2_billiard_gru/"
            "physics/evaluation_results_test.json"
        ),
    },
    {
        "environment": "billiard",
        "model_type": "latent_flow",
        "run": "envv2_billiard_latent_flow",
        "path": (
            "checkpoints/envv2_billiard_latent_flow/"
            "physics/evaluation_results_test.json"
        ),
    },
]


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_nested(
    data,
    keys,
    default=None,
):
    value = data

    for key in keys:
        if (
            not isinstance(value, dict)
            or key not in value
        ):
            return default

        value = value[key]

    return value


def normalize_path(value):
    if value is None:
        return None

    return os.path.normcase(
        os.path.normpath(
            str(value)
        )
    )


def extract_row(
    config,
    data,
):
    metadata = data.get(
        "metadata",
        {},
    )

    one_step = get_nested(
        data,
        [
            "one_step",
            "from_observed",
        ],
        {},
    )

    rollout = get_nested(
        data,
        [
            "rollout",
            "from_observed",
            "aggregate",
        ],
        {},
    )

    required = {
        "one_step_position_aee": (
            one_step.get(
                "position_aee"
            )
        ),
        "one_step_position_failures": (
            one_step.get(
                "position_failures"
            )
        ),
        "one_step_position_total": (
            one_step.get(
                "position_total"
            )
        ),
        "rollout_position_aee": (
            rollout.get(
                "position_aee"
            )
        ),
        "rollout_position_failures": (
            rollout.get(
                "position_failures"
            )
        ),
        "rollout_position_total": (
            rollout.get(
                "position_total"
            )
        ),
        "data_dir": metadata.get(
            "data_dir"
        ),
        "context": metadata.get(
            "context"
        ),
        "rollout_steps": metadata.get(
            "rollout_steps"
        ),
        "eval_stride": metadata.get(
            "eval_stride"
        ),
        "num_objects": metadata.get(
            "num_objects"
        ),
    }

    missing = [
        key
        for key, value in required.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            f"Result file for {config['run']} "
            "is missing fields: "
            + ", ".join(missing)
        )

    return {
        "environment": config[
            "environment"
        ],
        "model_type": config[
            "model_type"
        ],
        "run": config[
            "run"
        ],
        **required,
    }


def validate_environment_protocol(
    rows,
):
    errors = []

    comparable_fields = [
        "data_dir",
        "context",
        "rollout_steps",
        "eval_stride",
        "num_objects",
        "one_step_position_total",
        "rollout_position_total",
    ]

    environments = sorted(
        {
            row["environment"]
            for row in rows
        }
    )

    for environment in environments:
        env_rows = [
            row
            for row in rows
            if row["environment"] == environment
        ]

        if len(env_rows) != 3:
            errors.append(
                f"{environment}: expected 3 model "
                f"rows, found {len(env_rows)}"
            )
            continue

        for field in comparable_fields:
            values = []

            for row in env_rows:
                value = row[field]

                if field == "data_dir":
                    value = normalize_path(
                        value
                    )

                values.append(
                    value
                )

            if len(set(values)) != 1:
                details = ", ".join(
                    f"{row['model_type']}="
                    f"{row[field]}"
                    for row in env_rows
                )

                errors.append(
                    f"{environment}: "
                    f"mismatched {field}: "
                    f"{details}"
                )

    if errors:
        raise ValueError(
            "The environment results do not share "
            "one evaluation protocol:\n- "
            + "\n- ".join(errors)
        )


def collect_rows(
    root_dir,
):
    root = Path(
        root_dir
    )

    rows = []
    missing_files = []

    for config in RUNS:
        path = root / config["path"]

        if not path.exists():
            missing_files.append(
                str(path)
            )
            continue

        data = load_json(
            path
        )

        rows.append(
            extract_row(
                config,
                data,
            )
        )

    env_order = {
        "baseline": 0,
        "magnetic_wells": 1,
        "billiard": 2,
    }

    model_order = {
        "lstm": 0,
        "gru": 1,
        "latent_flow": 2,
    }

    rows.sort(
        key=lambda row: (
            env_order.get(
                row["environment"],
                999,
            ),
            model_order.get(
                row["model_type"],
                999,
            ),
        )
    )

    return (
        rows,
        missing_files,
    )


def format_float(value):
    return f"{float(value):.4f}"


def format_failures(
    failures,
    total,
):
    return (
        f"{int(failures)} / "
        f"{int(total)}"
    )


def write_csv(
    path,
    rows,
):
    fieldnames = [
        "environment",
        "model_type",
        "run",
        "one_step_position_aee",
        "one_step_position_failures",
        "one_step_position_total",
        "rollout_position_aee",
        "rollout_position_failures",
        "rollout_position_total",
        "data_dir",
        "context",
        "rollout_steps",
        "eval_stride",
        "num_objects",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_markdown(
    path,
    rows,
):
    if not rows:
        raise ValueError(
            "No result rows were collected"
        )

    context = rows[0][
        "context"
    ]

    rollout_steps = rows[0][
        "rollout_steps"
    ]

    eval_stride = rows[0][
        "eval_stride"
    ]

    lines = [
        "# Environment evaluation summary",
        "",
        (
            "All models were retrained on "
            "environment-specific shared "
            "train/validation splits and evaluated "
            "on the same held-out trajectories "
            "within each environment."
        ),
        "",
        (
            f"Common observed-space protocol: "
            f"context `{context}`, "
            f"rollout horizon `{rollout_steps}`, "
            f"evaluation stride `{eval_stride}`, "
            "and the same robust "
            "bright-foreground extractor."
        ),
        "",
        (
            "Only position AEE is used for direct "
            "model comparison. Lower is better."
        ),
        "",
        (
            "| Environment | Model | Run | "
            "1-step Pos AEE (↓) | "
            "1-step failures | "
            "Rollout Pos AEE (↓) | "
            "Rollout failures |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|"
        ),
    ]

    for row in rows:
        lines.append(
            f"| {row['environment']} "
            f"| {row['model_type']} "
            f"| {row['run']} "
            f"| {format_float(row['one_step_position_aee'])} "
            f"| {format_failures(row['one_step_position_failures'], row['one_step_position_total'])} "
            f"| {format_float(row['rollout_position_aee'])} "
            f"| {format_failures(row['rollout_position_failures'], row['rollout_position_total'])} |"
        )

    lines.extend(
        [
            "",
            (
                "Latent Flow uses auxiliary "
                "ground-truth state supervision "
                "during training; LSTM and GRU "
                "are trained from image "
                "reconstruction only."
            ),
            "",
            (
                "The billiard dataset used for "
                "this table must be the regenerated "
                "version with randomized, "
                "non-overlapping initial positions "
                "and randomized velocities."
            ),
        ]
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines) + "\n"
        )


def main(args):
    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    (
        rows,
        missing_files,
    ) = collect_rows(
        args.root_dir
    )

    if missing_files:
        missing_path = os.path.join(
            args.out_dir,
            "env_summary_missing_files.txt",
        )

        with open(
            missing_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "\n".join(
                    missing_files
                ) + "\n"
            )

        raise FileNotFoundError(
            f"Missing {len(missing_files)} "
            "result files. See: "
            f"{missing_path}"
        )

    validate_environment_protocol(
        rows
    )

    csv_path = os.path.join(
        args.out_dir,
        "env_summary.csv",
    )

    markdown_path = os.path.join(
        args.out_dir,
        "env_summary.md",
    )

    write_csv(
        csv_path,
        rows,
    )

    write_markdown(
        markdown_path,
        rows,
    )

    print(
        f"Saved {len(rows)} rows:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {markdown_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build the unified environment "
            "comparison table"
        )
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
    )

    main(
        parser.parse_args()
    )
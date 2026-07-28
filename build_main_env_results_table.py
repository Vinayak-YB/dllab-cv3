import argparse
import csv
import json
import os
from pathlib import Path


# This table combines two JSON schemas:
# 1) RNN baseline evaluations from main branch:
#       frame/from_observed and trajectory/from_observed
# 2) Latent-flow evaluations from eval_latent_flow.py:
#       one_step/from_observed and rollout/from_observed/aggregate
RUNS = [
    # Baseline / normal bouncing ball
    {
        "environment": "baseline",
        "model_type": "lstm",
        "run": "lstm_0",
        "path": "models/lstm_0/physics/evaluation_results.json",
        "format": "rnn_baseline",
    },
    {
        "environment": "baseline",
        "model_type": "lstm",
        "run": "lstm_1",
        "path": "models/lstm_1/physics/evaluation_results.json",
        "format": "rnn_baseline",
    },
    {
        "environment": "baseline",
        "model_type": "gru",
        "run": "gru_0",
        "path": "models/gru_0/physics/evaluation_results.json",
        "format": "rnn_baseline",
    },
    {
        "environment": "baseline",
        "model_type": "latent_flow",
        "run": "latent_flow_baseline_genloss",
        "path": "checkpoints/latent_flow_baseline_genloss/physics/evaluation_results_physics-data-v3.json",
        "format": "latent_flow",
    },

    # Magnetic wells
    {
        "environment": "magnetic_wells",
        "model_type": "lstm",
        "run": "lstm_2",
        "path": "models/lstm_2/physics/evaluation_results.json",
        "format": "rnn_baseline",
    },
    {
        "environment": "magnetic_wells",
        "model_type": "gru",
        "run": "gru_2",
        "path": "models/gru_2/physics/evaluation_results.json",
        "format": "rnn_baseline",
    },
    {
        "environment": "magnetic_wells",
        "model_type": "latent_flow",
        "run": "latent_flow_magwells_genloss",
        "path": "checkpoints/latent_flow_magwells_genloss/physics/evaluation_results_physics-data-magnetic_wells.json",
        "format": "latent_flow",
    },
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data, keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_metrics(data, fmt):
    if fmt == "rnn_baseline":
        return {
            "one_step_position_aee": get_nested(data, ["frame", "from_observed", "position_aee"]),
            "one_step_velocity_aee": get_nested(data, ["frame", "from_observed", "velocity_aee"]),
            "rollout_position_aee": get_nested(data, ["trajectory", "from_observed", "position_aee"]),
            "rollout_velocity_aee": get_nested(data, ["trajectory", "from_observed", "velocity_aee"]),
            "one_step_position_failures": get_nested(data, ["frame", "from_observed", "position_failures"]),
            "rollout_position_failures": get_nested(data, ["trajectory", "from_observed", "position_failures"]),
        }

    if fmt == "latent_flow":
        return {
            "one_step_position_aee": get_nested(data, ["one_step", "from_observed", "position_aee"]),
            "one_step_velocity_aee": get_nested(data, ["one_step", "from_observed", "velocity_aee"]),
            "rollout_position_aee": get_nested(data, ["rollout", "from_observed", "aggregate", "position_aee"]),
            "rollout_velocity_aee": get_nested(data, ["rollout", "from_observed", "aggregate", "velocity_aee"]),
            "one_step_position_failures": get_nested(data, ["one_step", "from_observed", "position_failures"]),
            "rollout_position_failures": get_nested(data, ["rollout", "from_observed", "aggregate", "position_failures"]),
        }

    raise ValueError(f"Unknown result format: {fmt}")


def format_float(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def collect_rows(root_dir):
    root = Path(root_dir)
    rows = []
    missing = []

    for cfg in RUNS:
        path = root / cfg["path"]
        if not path.exists():
            missing.append(str(path))
            continue

        data = load_json(path)
        metrics = extract_metrics(data, cfg["format"])
        rows.append({
            "environment": cfg["environment"],
            "model_type": cfg["model_type"],
            "run": cfg["run"],
            **metrics,
        })

    env_order = {"baseline": 0, "magnetic_wells": 1, "billiard": 2}
    model_order = {"lstm": 0, "gru": 1, "latent_flow": 2}
    rows.sort(key=lambda r: (
        env_order.get(r["environment"], 999),
        model_order.get(r["model_type"], 999),
        r["run"],
    ))
    return rows, missing


def write_csv(path, rows):
    fieldnames = [
        "environment",
        "model_type",
        "run",
        "one_step_position_aee",
        "one_step_velocity_aee",
        "rollout_position_aee",
        "rollout_velocity_aee",
        "one_step_position_failures",
        "rollout_position_failures",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = []
    lines.append("# Environment evaluation summary")
    lines.append("")
    lines.append("Comparable observed-space AEE for recurrent baselines and latent flow.")
    lines.append("")
    lines.append("| Environment | Model | Run | 1-step Pos AEE | 1-step Vel AEE | Rollout Pos AEE | Rollout Vel AEE |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['environment']} | {r['model_type']} | {r['run']} | "
            f"{format_float(r['one_step_position_aee'])} | "
            f"{format_float(r['one_step_velocity_aee'])} | "
            f"{format_float(r['rollout_position_aee'])} | "
            f"{format_float(r['rollout_velocity_aee'])} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    rows, missing = collect_rows(args.root_dir)

    csv_path = os.path.join(args.out_dir, "env_summary.csv")
    md_path = os.path.join(args.out_dir, "env_summary.md")
    missing_path = os.path.join(args.out_dir, "env_summary_missing_files.txt")

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    if missing:
        with open(missing_path, "w", encoding="utf-8") as f:
            for path in missing:
                f.write(path + "\n")

    print(f"Saved {len(rows)} rows:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    if missing:
        print(f"Missing {len(missing)} files. See:")
        print(f"  {missing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
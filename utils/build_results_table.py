import argparse
import json
import os
import csv


SPLIT_FILES = [
    "physics/evaluation_results_test_id.json",
    "physics/evaluation_results_test_ood_gravity.json",
    "physics/evaluation_results_test_ood_velocity.json",
    "physics/evaluation_results_test_ood_position.json",
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_split_name(data, fallback_path):
    meta = data.get("metadata", {})
    data_dir = meta.get("data_dir", "")
    if data_dir:
        return os.path.basename(os.path.normpath(data_dir))
    filename = os.path.basename(fallback_path)
    return filename.replace("evaluation_results_", "").replace(".json", "")


def get_nested(data, keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_recurrent_metrics(data):
    return {
        "one_step_position_aee": data["one_step"]["from_observed"]["position_aee"],
        "one_step_velocity_aee": data["one_step"]["from_observed"]["velocity_aee"],
        "rollout_position_aee": data["rollout"]["from_observed"]["aggregate"]["position_aee"],
        "rollout_velocity_aee": data["rollout"]["from_observed"]["aggregate"]["velocity_aee"],
        "one_step_position_failures": data["one_step"]["from_observed"].get("position_failures"),
        "one_step_velocity_failures": data["one_step"]["from_observed"].get("velocity_failures"),
        "rollout_position_failures": data["rollout"]["from_observed"]["aggregate"].get("position_failures"),
        "rollout_velocity_failures": data["rollout"]["from_observed"]["aggregate"].get("velocity_failures"),
        "latent_one_step_position_r2": data["one_step"].get("from_latent", {}).get("position_r2"),
        "latent_one_step_velocity_r2": data["one_step"].get("from_latent", {}).get("velocity_r2"),
        "latent_rollout_position_aee": data["rollout"].get("from_latent", {}).get("aggregate", {}).get("position_aee"),
        "latent_rollout_velocity_aee": data["rollout"].get("from_latent", {}).get("aggregate", {}).get("velocity_aee"),
    }


def extract_state_metrics(data):
    return {
        "one_step_position_aee": data["one_step"]["position_aee"],
        "one_step_velocity_aee": data["one_step"]["velocity_aee"],
        "one_step_position_r2": data["one_step"].get("position_r2"),
        "one_step_velocity_r2": data["one_step"].get("velocity_r2"),
        "rollout_position_aee": data["rollout"]["aggregate"]["position_aee"],
        "rollout_velocity_aee": data["rollout"]["aggregate"]["velocity_aee"],
        "rollout_position_r2": data["rollout"]["aggregate"].get("position_r2"),
        "rollout_velocity_r2": data["rollout"]["aggregate"].get("velocity_r2"),
    }


def extract_latent_flow_metrics(data):
    """Extract latent-flow metrics.

    Comparable metrics use observed-space AEE, matching the recurrent extractor:
    one_step/from_observed and rollout/from_observed/aggregate.

    Additional latent-specific metrics are kept in evaluation_summary_all_metrics.csv:
    - from_latent_probe
    - from_state_head
    """
    return {
        # Comparable observed-space metrics
        "one_step_position_aee": data["one_step"]["from_observed"]["position_aee"],
        "one_step_velocity_aee": data["one_step"]["from_observed"]["velocity_aee"],
        "rollout_position_aee": data["rollout"]["from_observed"]["aggregate"]["position_aee"],
        "rollout_velocity_aee": data["rollout"]["from_observed"]["aggregate"]["velocity_aee"],
        "one_step_position_failures": data["one_step"]["from_observed"].get("position_failures"),
        "one_step_velocity_failures": data["one_step"]["from_observed"].get("velocity_failures"),
        "rollout_position_failures": data["rollout"]["from_observed"]["aggregate"].get("position_failures"),
        "rollout_velocity_failures": data["rollout"]["from_observed"]["aggregate"].get("velocity_failures"),

        # Latent probe one-step metrics
        "latent_probe_one_step_position_r2": get_nested(data, ["one_step", "from_latent_probe", "position_r2"]),
        "latent_probe_one_step_velocity_r2": get_nested(data, ["one_step", "from_latent_probe", "velocity_r2"]),
        "latent_probe_one_step_position_aee": get_nested(data, ["one_step", "from_latent_probe", "position_aee"]),
        "latent_probe_one_step_velocity_aee": get_nested(data, ["one_step", "from_latent_probe", "velocity_aee"]),
        "latent_probe_rollout_position_aee": get_nested(data, ["rollout", "from_latent_probe", "aggregate", "position_aee"]),
        "latent_probe_rollout_velocity_aee": get_nested(data, ["rollout", "from_latent_probe", "aggregate", "velocity_aee"]),

        # State-head one-step metrics
        "state_head_one_step_position_r2": get_nested(data, ["one_step", "from_state_head", "position_r2"]),
        "state_head_one_step_velocity_r2": get_nested(data, ["one_step", "from_state_head", "velocity_r2"]),
        "state_head_one_step_position_aee": get_nested(data, ["one_step", "from_state_head", "position_aee"]),
        "state_head_one_step_velocity_aee": get_nested(data, ["one_step", "from_state_head", "velocity_aee"]),
        "state_head_rollout_position_aee": get_nested(data, ["rollout", "from_state_head", "aggregate", "position_aee"]),
        "state_head_rollout_velocity_aee": get_nested(data, ["rollout", "from_state_head", "aggregate", "velocity_aee"]),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def write_markdown(path, rows):
    lines = []
    lines.append("# Evaluation summary\n")
    lines.append("Comparable metrics across models use observed-space AEE for one-step and rollout.\n")
    lines.append("| Model | Run | Split | 1-step Pos AEE | 1-step Vel AEE | Rollout Pos AEE | Rollout Vel AEE |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['model_type']} | {r['model_run']} | {r['split']} | "
            f"{format_float(r['one_step_position_aee'])} | {format_float(r['one_step_velocity_aee'])} | "
            f"{format_float(r['rollout_position_aee'])} | {format_float(r['rollout_velocity_aee'])} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    configs = {
        "recurrent_id_v2": {
            "model_type": "recurrent",
            "files": SPLIT_FILES,
        },
        "state_mlp_v2": {
            "model_type": "state_mlp",
            "files": SPLIT_FILES,
        },

        # New g=-3 environment runs
        "recurrent_g3": {
            "model_type": "recurrent",
            "files": SPLIT_FILES,
        },
        "state_mlp_g3": {
            "model_type": "state_mlp",
            "files": SPLIT_FILES,
        },
        "latent_flow_g3_genloss": {
            "model_type": "latent_flow",
            "files": SPLIT_FILES,
        },
    }

    split_order = {
        "test_id": 0,
        "test_ood_gravity": 1,
        "test_ood_velocity": 2,
        "test_ood_position": 3,
    }
    model_order = {
        "recurrent": 0,
        "state_mlp": 1,
        "latent_flow": 2,
    }

    all_rows = []
    comparable_rows = []
    missing_files = []

    for run_name, cfg in configs.items():
        model_dir = os.path.join(args.checkpoints_dir, run_name)

        for rel_path in cfg["files"]:
            json_path = os.path.join(model_dir, rel_path)
            if not os.path.exists(json_path):
                missing_files.append(json_path)
                continue

            data = load_json(json_path)
            split = extract_split_name(data, json_path)

            row = {
                "model_run": run_name,
                "model_type": cfg["model_type"],
                "split": split,
            }

            if cfg["model_type"] == "recurrent":
                row.update(extract_recurrent_metrics(data))
            elif cfg["model_type"] == "state_mlp":
                row.update(extract_state_metrics(data))
            elif cfg["model_type"] == "latent_flow":
                row.update(extract_latent_flow_metrics(data))
            else:
                continue

            all_rows.append(row)

            comparable_rows.append(
                {
                    "model_type": row["model_type"],
                    "model_run": row["model_run"],
                    "split": row["split"],
                    "one_step_position_aee": row["one_step_position_aee"],
                    "one_step_velocity_aee": row["one_step_velocity_aee"],
                    "rollout_position_aee": row["rollout_position_aee"],
                    "rollout_velocity_aee": row["rollout_velocity_aee"],
                }
            )

    all_rows.sort(key=lambda r: (model_order.get(r["model_type"], 999), r["model_run"], split_order.get(r["split"], 999)))
    comparable_rows.sort(key=lambda r: (model_order.get(r["model_type"], 999), r["model_run"], split_order.get(r["split"], 999)))

    if all_rows:
        all_fieldnames = sorted({k for row in all_rows for k in row.keys()})
        write_csv(os.path.join(args.out_dir, "evaluation_summary_all_metrics.csv"), all_rows, all_fieldnames)

    if comparable_rows:
        cmp_fieldnames = [
            "model_type",
            "model_run",
            "split",
            "one_step_position_aee",
            "one_step_velocity_aee",
            "rollout_position_aee",
            "rollout_velocity_aee",
        ]
        write_csv(os.path.join(args.out_dir, "evaluation_summary_comparable_metrics.csv"), comparable_rows, cmp_fieldnames)
        write_markdown(os.path.join(args.out_dir, "evaluation_summary.md"), comparable_rows)

    if missing_files:
        with open(os.path.join(args.out_dir, "missing_files.txt"), "w") as f:
            for path in missing_files:
                f.write(path + "\n")

    print(f"Saved results in: {args.out_dir}")
    print(f"Found {len(all_rows)} result files.")
    if missing_files:
        print(f"Missing {len(missing_files)} files. See missing_files.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build summary tables from evaluation JSON files")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)

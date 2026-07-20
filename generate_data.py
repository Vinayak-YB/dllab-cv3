import argparse
import json
import os
import shutil

import numpy as np
import tqdm
from PIL import Image, ImageDraw


def render_circle(width, height, x, y, radius, filename):
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    left = x - radius
    top = height - (y + radius)
    right = x + radius
    bottom = height - (y - radius)

    draw.ellipse([left, top, right, bottom], fill="blue")
    img.save(filename)


def ensure_clean_dir(path, overwrite=False):
    if overwrite and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def sample_initial_state(width, height, radius, velocity_scale, position_mode="id"):
    mode = np.random.choice(
        ["drop", "drop+horizontal", "parabolic", "parabolic+biased"],
        p=[0.2, 0.2, 0.3, 0.3]
    )

    if position_mode == "id":
        s_x = np.random.randint(radius, width - radius)
        s_y = np.random.randint(int(0.3 * height), height - radius)
    elif position_mode == "ood":
        edge_band = 15

        if np.random.rand() < 0.5:
            s_x = np.random.randint(radius, radius + edge_band)
        else:
            s_x = np.random.randint(width - radius - edge_band, width - radius)

        if np.random.rand() < 0.5:
            s_y = np.random.randint(radius, radius + edge_band)
        else:
            s_y = np.random.randint(height - radius - edge_band, height - radius)
    else:
        raise ValueError(f"Unknown position_mode: {position_mode}")

    if mode == "drop":
        v_x, v_y = 0.0, 0.0
    elif mode == "drop+horizontal":
        v_x, v_y = np.random.randn() * velocity_scale, 0.0
    elif mode == "parabolic":
        v_x, v_y = np.random.randn() * velocity_scale, np.random.randn() * velocity_scale
    elif mode == "parabolic+biased":
        v_x = np.random.randn() * velocity_scale
        v_y = np.random.randn() * velocity_scale + velocity_scale

    return mode, float(s_x), float(s_y), float(v_x), float(v_y)


def simulate_trajectory(
    width,
    height,
    radius,
    gravity,
    max_traj_length,
    velocity_scale,
    position_mode="id",
):
    mode, s_x, s_y, v_x, v_y = sample_initial_state(
        width=width,
        height=height,
        radius=radius,
        velocity_scale=velocity_scale,
        position_mode=position_mode,
    )

    positions = []
    velocities = []

    stopped_for = 0
    num_bounces_floor = 0
    num_bounces_wall = 0
    num_bounces_top = 0

    initial_state = {
        "mode": mode,
        "x": s_x,
        "y": s_y,
        "vx": v_x,
        "vy": v_y,
    }

    for _ in range(max_traj_length):
        positions.append((s_x, s_y))
        velocities.append((v_x, v_y))

        v_y += gravity
        s_y += v_y
        s_x += v_x

        if s_y - radius <= 0:
            s_y = radius
            v_y *= -0.7
            v_x *= 0.9
            v_y = v_y if np.abs(v_y) >= 2 else 0.0
            v_x = v_x if np.abs(v_x) >= 1 else 0.0
            stopped_for += int(v_y == 0.0 and v_x == 0.0)
            num_bounces_floor += 1

        if s_y + radius >= height:
            s_y = height - radius
            v_y *= -0.7
            num_bounces_top += 1

        if s_x - radius <= 0:
            s_x = radius
            v_x *= -0.7
            num_bounces_wall += 1

        if s_x + radius >= width:
            s_x = width - radius
            v_x *= -0.7
            num_bounces_wall += 1

        if stopped_for > 5:
            break

    trajectory_info = {
        "initial_state": initial_state,
        "num_frames": len(positions),
        "stopped_early": stopped_for > 5,
        "num_bounces_floor": num_bounces_floor,
        "num_bounces_wall": num_bounces_wall,
        "num_bounces_top": num_bounces_top,
    }

    return (
        np.array(positions, dtype=np.float32),
        np.array(velocities, dtype=np.float32),
        trajectory_info,
    )


def write_split(
    root_dir,
    split_name,
    n_trajectories,
    gravity,
    width,
    height,
    radius,
    max_traj_length,
    velocity_scale,
    position_mode,
    seed,
    overwrite=False,
):
    split_dir = os.path.join(root_dir, split_name)
    ensure_clean_dir(split_dir, overwrite=overwrite)

    metadata = {
        "dataset_name": "bouncing_ball",
        "split_name": split_name,
        "n_trajectories": n_trajectories,
        "gravity": gravity,
        "width": width,
        "height": height,
        "radius": radius,
        "max_traj_length": max_traj_length,
        "velocity_scale": velocity_scale,
        "position_mode": position_mode,
        "seed": seed,
    }

    with open(os.path.join(split_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    for i in tqdm.tqdm(range(n_trajectories), desc=f"Generating {split_name}"):
        traj_dir = os.path.join(split_dir, f"traj-{i:05d}")
        os.makedirs(traj_dir, exist_ok=True)

        positions, velocities, traj_info = simulate_trajectory(
            width=width,
            height=height,
            radius=radius,
            gravity=gravity,
            max_traj_length=max_traj_length,
            velocity_scale=velocity_scale,
            position_mode=position_mode,
        )

        for frame_idx, (x, y) in enumerate(positions):
            render_circle(
                width=width,
                height=height,
                x=x,
                y=y,
                radius=radius,
                filename=os.path.join(traj_dir, f"frame_{frame_idx:03d}.png"),
            )

        np.save(os.path.join(traj_dir, "positions.npy"), positions)
        np.save(os.path.join(traj_dir, "velocities.npy"), velocities)

        with open(os.path.join(traj_dir, "trajectory.json"), "w") as f:
            json.dump(traj_info, f, indent=4)


def main(args):
    np.random.seed(args.seed)

    root_dir = os.path.join(args.data_root, "bouncing_ball")
    os.makedirs(root_dir, exist_ok=True)

    if args.overwrite_id:
        for split in ["train", "val", "test_id"]:
            split_dir = os.path.join(root_dir, split)
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)

    if args.overwrite_ood:
        for split in ["test_ood_gravity", "test_ood_velocity", "test_ood_position"]:
            split_dir = os.path.join(root_dir, split)
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)

    write_split(
        root_dir=root_dir,
        split_name="train",
        n_trajectories=args.n_train,
        gravity=args.id_gravity,
        width=args.width,
        height=args.height,
        radius=args.radius,
        max_traj_length=args.max_traj_length,
        velocity_scale=args.id_velocity_scale,
        position_mode="id",
        seed=args.seed,
        overwrite=False,
    )

    write_split(
        root_dir=root_dir,
        split_name="val",
        n_trajectories=args.n_val,
        gravity=args.id_gravity,
        width=args.width,
        height=args.height,
        radius=args.radius,
        max_traj_length=args.max_traj_length,
        velocity_scale=args.id_velocity_scale,
        position_mode="id",
        seed=args.seed,
        overwrite=False,
    )

    write_split(
        root_dir=root_dir,
        split_name="test_id",
        n_trajectories=args.n_test_id,
        gravity=args.id_gravity,
        width=args.width,
        height=args.height,
        radius=args.radius,
        max_traj_length=args.max_traj_length,
        velocity_scale=args.id_velocity_scale,
        position_mode="id",
        seed=args.seed,
        overwrite=False,
    )

    write_split(
        root_dir=root_dir,
        split_name="test_ood_gravity",
        n_trajectories=args.n_test_ood,
        gravity=args.ood_gravity,
        width=args.width,
        height=args.height,
        radius=args.radius,
        max_traj_length=args.max_traj_length,
        velocity_scale=args.id_velocity_scale,
        position_mode="id",
        seed=args.seed,
        overwrite=False,
    )

    write_split(
        root_dir=root_dir,
        split_name="test_ood_velocity",
        n_trajectories=args.n_test_ood,
        gravity=args.id_gravity,
        width=args.width,
        height=args.height,
        radius=args.radius,
        max_traj_length=args.max_traj_length,
        velocity_scale=args.ood_velocity_scale,
        position_mode="id",
        seed=args.seed,
        overwrite=False,
    )

    write_split(
        root_dir=root_dir,
        split_name="test_ood_position",
        n_trajectories=args.n_test_ood,
        gravity=args.id_gravity,
        width=args.width,
        height=args.height,
        radius=args.radius,
        max_traj_length=args.max_traj_length,
        velocity_scale=args.id_velocity_scale,
        position_mode="ood",
        seed=args.seed,
        overwrite=False,
    )

    print(f"\nDataset written to: {root_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate structured bouncing ball dataset with ID and OOD splits (loader-compatible)"
    )

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--max_traj_length", type=int, default=100)

    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test_id", type=int, default=100)
    parser.add_argument("--n_test_ood", type=int, default=100)

    parser.add_argument("--id_gravity", type=float, default=-1.0)
    parser.add_argument("--ood_gravity", type=float, default=-2.0)

    parser.add_argument("--id_velocity_scale", type=float, default=5.0)
    parser.add_argument("--ood_velocity_scale", type=float, default=9.0)

    parser.add_argument("--overwrite_id", action="store_true")
    parser.add_argument("--overwrite_ood", action="store_true")

    args = parser.parse_args()
    main(args)
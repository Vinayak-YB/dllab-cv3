import os
import argparse
import numpy as np
import cv2
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')  # Use non-interactive backend


def apply_position_shift(pos, position_shift, img_size, radius=8):
    """Applies spawn position distribution shifts for OOD evaluations."""
    if position_shift == "corners":
        # Force initial position near the top-left or top-right corners
        corner_x = np.random.choice([radius + 5, img_size - radius - 5])
        corner_y = np.random.uniform(radius + 5, radius + 25)
        return np.array([corner_x, corner_y], dtype=float)
    elif position_shift == "top_half":
        # Force initial position to upper half of canvas
        return np.array([
            np.random.uniform(radius + 10, img_size - radius - 10),
            np.random.uniform(radius + 10, (img_size / 2) - 10)
        ], dtype=float)
    return pos


def simulate_bouncing_ball(num_frames=100, img_size=128, radius=8, gravity_scale=1.0, vel_scale=1.0, position_shift="none"):
    """Env 0: Standard Linear Bouncing Ball with Gravity & OOD Parameters"""
    pos = np.array([np.random.uniform(radius + 10, img_size - radius - 10),
                    np.random.uniform(radius + 10, img_size - radius - 10)], dtype=float)
    pos = apply_position_shift(pos, position_shift, img_size, radius)

    vel = np.random.uniform(-4, 4, size=2) * vel_scale
    gravity = np.array([0.0, 0.25]) * gravity_scale
    
    frames, positions, velocities = [], [], []
    for _ in range(num_frames):
        # Draw Frame
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.circle(img, (int(pos[0]), int(pos[1])), radius, 255, -1)
        
        frames.append(img)
        positions.append(pos.copy())
        velocities.append(vel.copy())
        
        # Physics Step
        vel += gravity
        pos += vel
        
        # Wall Collisions
        for i in range(2):
            if pos[i] - radius < 0:
                pos[i] = radius
                vel[i] *= -0.9
            elif pos[i] + radius > img_size:
                pos[i] = img_size - radius
                vel[i] *= -0.9
                
    return frames, np.array(positions), np.array(velocities)


def simulate_magnetic_wells(num_frames=100, img_size=128, radius=8, gravity_scale=1.0, vel_scale=1.0, position_shift="none"):
    """Env 1: Chaotic Orbit under 2 Attractors & OOD Parameters"""
    pos = np.array([np.random.uniform(20, img_size - 20), 
                    np.random.uniform(20, img_size - 20)], dtype=float)
    pos = apply_position_shift(pos, position_shift, img_size, radius)

    vel = np.random.uniform(-3, 3, size=2) * vel_scale
    
    # Fixed Attractor Coordinates
    wells = [np.array([40.0, 64.0]), np.array([88.0, 64.0])]
    G = 150.0 * gravity_scale  # Gravitational/Magnetic attraction strength scale
    
    frames, positions, velocities = [], [], []
    for _ in range(num_frames):
        # Draw Frame
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        for w in wells:
            cv2.circle(img, (int(w[0]), int(w[1])), 2, 100, -1)
        cv2.circle(img, (int(pos[0]), int(pos[1])), radius, 255, -1)
        
        frames.append(img)
        positions.append(pos.copy())
        velocities.append(vel.copy())
        
        # Compute Magnetic Acceleration Vectors
        acc = np.zeros(2)
        for w in wells:
            diff = w - pos
            dist = np.linalg.norm(diff) + 1e-3
            acc += (G / (dist**2)) * (diff / dist)
            
        vel += acc
        speed = np.linalg.norm(vel)
        max_speed = 8.0 * vel_scale
        if speed > max_speed:
            vel = (vel / speed) * max_speed
            
        pos += vel
        
        # Wall Bounds
        for i in range(2):
            if pos[i] - radius < 0:
                pos[i] = radius
                vel[i] *= -0.85
            elif pos[i] + radius > img_size:
                pos[i] = img_size - radius
                vel[i] *= -0.85
                
    return frames, np.array(positions), np.array(velocities)


def simulate_billiard_balls(num_frames=100, img_size=128, radius=8, gravity_scale=1.0, vel_scale=1.0, position_shift="none"):
    """Env 2: 2-Ball Elastic Collisions & OOD Parameters"""
    pos1 = np.array([30.0, 40.0], dtype=float)
    vel1 = np.array([3.5, 2.0], dtype=float) * vel_scale
    
    pos2 = np.array([90.0, 80.0], dtype=float)
    vel2 = np.array([-2.5, -3.0], dtype=float) * vel_scale

    if position_shift == "corners":
        pos1 = np.array([20.0, 20.0], dtype=float)
        pos2 = np.array([20.0, 50.0], dtype=float)
    elif position_shift == "top_half":
        pos1 = np.array([30.0, 30.0], dtype=float)
        pos2 = np.array([80.0, 40.0], dtype=float)

    # gravity_scale adds a downward force vector if specified (billiards default is 0 gravity)
    downward_force = np.array([0.0, 0.20 * (gravity_scale - 1.0)]) if gravity_scale != 1.0 else np.zeros(2)

    frames, positions, velocities = [], [], []
    for _ in range(num_frames):
        # Draw Frame (Ball 1 bright 255, Ball 2 180)
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.circle(img, (int(pos1[0]), int(pos1[1])), radius, 255, -1)
        cv2.circle(img, (int(pos2[0]), int(pos2[1])), radius, 180, -1)
        
        frames.append(img)
        positions.append(np.concatenate([pos1, pos2]))
        velocities.append(np.concatenate([vel1, vel2]))
        
        # Position & Force Update
        vel1 += downward_force
        vel2 += downward_force
        pos1 += vel1
        pos2 += vel2
        
        # Ball-to-Ball Elastic Collision
        diff = pos2 - pos1
        dist = np.linalg.norm(diff)
        if dist < (2 * radius):
            normal = diff / (dist + 1e-5)
            rel_vel = vel2 - vel1
            vel_along_normal = np.dot(rel_vel, normal)
            if vel_along_normal < 0:
                impulse = vel_along_normal * normal
                vel1 += impulse
                vel2 -= impulse
                
        # Wall Bounce Logic for both balls
        for p, v in [(pos1, vel1), (pos2, vel2)]:
            for i in range(2):
                if p[i] - radius < 0:
                    p[i] = radius
                    v[i] *= -0.9
                elif p[i] + radius > img_size:
                    p[i] = img_size - radius
                    v[i] *= -0.9
                    
    return frames, np.array(positions), np.array(velocities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='magnetic_wells', 
                        choices=['bouncing_ball', 'magnetic_wells', 'billiard'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_trajectories', type=int, default=110)
    parser.add_argument('--num_frames', type=int, default=100)
    parser.add_argument("--gravity_scale", type=float, default=1.0, help="Scale factor for gravity / force strength")
    parser.add_argument("--vel_scale", type=float, default=1.0, help="Scale factor for initial velocity")
    parser.add_argument("--position_shift", type=str, default="none", choices=["none", "corners", "top_half"], help="Shift initial spawn position distribution")
    args = parser.parse_args()

    out_dir = args.output_dir or f"physics-data-{args.env}"
    os.makedirs(out_dir, exist_ok=True)

    sim_func = {
        'bouncing_ball': simulate_bouncing_ball,
        'magnetic_wells': simulate_magnetic_wells,
        'billiard': simulate_billiard_balls
    }[args.env]

    print(f"Generating {args.num_trajectories} trajectories for Environment: [{args.env}]...")
    for idx in tqdm(range(args.num_trajectories)):
        traj_dir = os.path.join(out_dir, f"traj-{idx:03d}")
        os.makedirs(traj_dir, exist_ok=True)
        
        frames, positions, velocities = sim_func(
            num_frames=args.num_frames,
            gravity_scale=args.gravity_scale,
            vel_scale=args.vel_scale,
            position_shift=args.position_shift
        )
        
        # Save frame images as frame_000.png, frame_001.png, ...
        for f_idx, frame in enumerate(frames):
            cv2.imwrite(os.path.join(traj_dir, f"frame_{f_idx:03d}.png"), frame)
            
        # Save positions and velocities
        np.save(os.path.join(traj_dir, "positions.npy"), positions)
        np.save(os.path.join(traj_dir, "velocities.npy"), velocities)

    print(f"Done! Trajectories saved with frame PNGs to {out_dir}/")

if __name__ == "__main__":
    main()
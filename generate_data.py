import matplotlib.pyplot as plt
import numpy as np
import shutil
from tqdm import tqdm
import os
import matplotlib
import argparse
import cv2
matplotlib.use('Agg')  # Use non-interactive backend


def render_circle(width, height, x, y, radius, filename):
    # Create fixed-size figure
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)  # 1.28 inches * 100 dpi = 128 pixels
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire canvas
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_facecolor('white')
    ax.add_patch(plt.Circle((x, y), radius, color='blue'))
    ax.axis('off')

    fig.savefig(filename, dpi=100)
    plt.close(fig)
def simulate_bouncing_ball(num_frames=100, img_size=128, radius=8):
    """Env 0: Standard Linear Bouncing Ball with Gravity"""
    pos = np.array([np.random.uniform(radius + 10, img_size - radius - 10),
                    np.random.uniform(radius + 10, img_size - radius - 10)], dtype=float)
    vel = np.random.uniform(-4, 4, size=2)
    gravity = np.array([0.0, 0.25])
    
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


def simulate_magnetic_wells(num_frames=100, img_size=128, radius=8):
    """Env 1: Chaotic Orbit under 2 Gravity/Magnetic Attractors"""
    pos = np.array([np.random.uniform(20, img_size - 20), 
                    np.random.uniform(20, img_size - 20)], dtype=float)
    vel = np.random.uniform(-3, 3, size=2)
    
    # Fixed Attractor Coordinates
    wells = [np.array([40.0, 64.0]), np.array([88.0, 64.0])]
    G = 150.0  # Gravitational constant scale
    
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
        if speed > 8.0:
            vel = (vel / speed) * 8.0
            
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


def simulate_billiard_balls(num_frames=100, img_size=128, radius=8):
    """Env 2: 2-Ball Elastic Collisions"""
    pos1 = np.array([30.0, 40.0], dtype=float)
    vel1 = np.array([3.5, 2.0], dtype=float)
    
    pos2 = np.array([90.0, 80.0], dtype=float)
    vel2 = np.array([-2.5, -3.0], dtype=float)
    
    frames, positions, velocities = [], [], []
    for _ in range(num_frames):
        # Draw Frame (Ball 1 bright 255, Ball 2 180)
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.circle(img, (int(pos1[0]), int(pos1[1])), radius, 255, -1)
        cv2.circle(img, (int(pos2[0]), int(pos2[1])), radius, 180, -1)
        
        frames.append(img)
        positions.append(np.concatenate([pos1, pos2]))
        velocities.append(np.concatenate([vel1, vel2]))
        
        # Position Update
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
        
        frames, positions, velocities = sim_func(num_frames=args.num_frames)
        
        # Save frame images as frame_000.png, frame_001.png, ...
        for f_idx, frame in enumerate(frames):
            cv2.imwrite(os.path.join(traj_dir, f"frame_{f_idx:03d}.png"), frame)
            
        # Save positions and velocities
        np.save(os.path.join(traj_dir, "positions.npy"), positions)
        np.save(os.path.join(traj_dir, "velocities.npy"), velocities)

    print(f"Done! Trajectories saved with frame PNGs to {out_dir}/")

if __name__ == "__main__":
    main()
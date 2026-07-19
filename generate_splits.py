import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

def render_circle(width, height, x, y, radius, filename):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_facecolor('white')
    ax.add_patch(plt.Circle((x, y), radius, color='blue'))
    ax.axis('off')
    fig.savefig(filename, dpi=100)
    plt.close(fig)


def generate_split(split_name, n_trajectories=250, gravity=-1.0, vel_noise=5.0, pos_bias=0):
    save_dir = f"data_{split_name}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Fixed different seed per split
    seed = 42 + {"id":0, "ood_gravity":1, "ood_velocity":2, "ood_position":3}[split_name]
    np.random.seed(seed)

    width, height = 128, 128
    radius = 5
    max_traj_length = 100

    for i in tqdm(range(n_trajectories), desc=f'Generating {split_name}'):
        traj_dir = os.path.join(save_dir, f'traj-{i}')
        os.makedirs(traj_dir, exist_ok=True)

        s_x = np.random.randint(20, width-20) + pos_bias
        s_y = np.random.randint(int(0.3*height), height-20)

        v_x = np.random.randn() * vel_noise
        v_y = np.random.randn() * vel_noise

        positions = []
        velocities = []

        for frame in range(max_traj_length):
            render_circle(width, height, s_x, s_y, radius, 
                         os.path.join(traj_dir, f'frame_{frame:03}.png'))
            positions.append([s_x, s_y])
            velocities.append([v_x, v_y])

            v_y += gravity
            s_y += v_y
            s_x += v_x

            if s_y - radius <= 0:
                s_y = radius
                v_y *= -0.7
                v_x *= 0.9
                if abs(v_y) < 2: v_y = 0
                if abs(v_x) < 1: v_x = 0

            if s_y + radius >= height:
                s_y = height - radius
                v_y *= -0.7
            if s_x - radius <= 0 or s_x + radius >= width:
                s_x = np.clip(s_x, radius, width - radius)
                v_x *= -0.7

        np.save(os.path.join(traj_dir, "positions.npy"), np.array(positions))
        np.save(os.path.join(traj_dir, "velocities.npy"), np.array(velocities))

if __name__ == "__main__":
    print("Generating 4 data splits...")
    generate_split("id", n_trajectories=250)
    generate_split("ood_gravity", n_trajectories=250, gravity=-1.8)
    generate_split("ood_velocity", n_trajectories=250, vel_noise=12.0)
    generate_split("ood_position", n_trajectories=250, pos_bias=40)
    print("All splits generated!")
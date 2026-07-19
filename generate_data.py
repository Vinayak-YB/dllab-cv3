import matplotlib.pyplot as plt
import numpy as np
import shutil
import tqdm
import os
import matplotlib
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


if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility

    # Settings
    width, height = 128, 128
    save_dir = "physics-data-v3"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  # Clear previous data

    radius = 5
    a = -1
    max_traj_length = 100

    n_trajectories = 1000
    for i in tqdm.tqdm(range(n_trajectories), desc='Generating trajectories'):
        os.makedirs(os.path.join(save_dir, f'traj-{i}'), exist_ok=True)
        positions = []
        velocities = []

        # Initial object state
        mode = np.random.choice(['drop', 'drop+horizontal', 'parabolic', 'parabolic+biased'], p= [0.2, 0.2, 0.3, 0.3])
        s_x = np.random.randint(radius, width - radius)
        s_y = np.random.randint(int(0.3 * height), height - radius)

        if mode == 'drop':
            v_x = 0
            v_y = 0
        elif mode == 'drop+horizontal':
            v_x = np.random.randn() * 5
            v_y = 0
        elif mode == 'parabolic':
            v_x = np.random.randn() * 5
            v_y = np.random.randn() * 5
        elif mode == 'parabolic+biased':
            v_x = np.random.randn() * 5
            v_y = np.random.randn() * 5 + 5
        else:
            raise ValueError(f'Unknown mode: {mode}')

        stopped_for = 0
        for frame in range(max_traj_length):
            # create image from current state
            render_circle(width, height, s_x, s_y, radius, os.path.join(save_dir, f'traj-{i}', f'frame_{frame:03}.png'))
            positions.append((s_x, s_y))
            velocities.append((v_x, v_y))

            # update state
            v_y += a
            s_y += v_y
            s_x += v_x

            # Bounce off the floor
            if s_y - radius <= 0:
                s_y = radius
                v_y *= -0.7

                # Apply friction to horizontal velocity
                v_x *= 0.9

                # Clip low velocities to zero
                v_y = v_y if np.abs(v_y) >= 2 else 0
                v_x = v_x if np.abs(v_x) >= 1 else 0
                
                stopped_for += (v_y == 0 and v_x == 0)

            # Bounce off the ceiling
            if s_y + radius >= height:
                s_y = height - radius
                v_y *= -0.7

            # Bounce off left wall
            if s_x - radius <= 0:
                s_x = radius
                v_x *= -0.7

            # Bounce off right wall
            if s_x + radius >= width:
                s_x = width - radius
                v_x *= -0.7

            if stopped_for > 5:
                break

        # Save positions and velocities
        np.save(os.path.join(save_dir, f"traj-{i}", "positions.npy"), np.array(positions))
        np.save(os.path.join(save_dir, f"traj-{i}", "velocities.npy"), np.array(velocities))

import sys
import json
import os
import matplotlib.pyplot as plt

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'evaluation_results.json'

    if not os.path.exists(json_path):
        print(f"Error: File '{json_path}' not found.")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    modes = ['frame', 'trajectory']
    pos_aee = [data.get(m, {}).get('from_observed', {}).get('position_aee', 0) for m in modes]
    failures = [data.get(m, {}).get('from_observed', {}).get('position_failures', 0) for m in modes]
    pos_r2 = [max(0, data.get(m, {}).get('from_latent', {}).get('position_r2', 0)) for m in modes]
    vel_r2 = [max(0, data.get(m, {}).get('from_latent', {}).get('velocity_r2', 0)) for m in modes]

    x_labels = ['Single Frame', 'Trajectory Unroll']

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Helper function to style line plots cleanly
    def plot_line(ax, data_y, title, ylabel, ylim=None):
        ax.plot(x_labels, data_y, marker='o', linewidth=2.5, markersize=8, color='#d95f02')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        if ylim:
            ax.set_ylim(ylim)

    # 1. Position Error
    plot_line(axes[0, 0], pos_aee, 'Position Error (px - Lower is better)', 'Pixels')

    # 2. Tracking Failures
    plot_line(axes[0, 1], failures, 'Failures Count (Lower is better)', 'Failures')

    # 3. Latent Position R2
    plot_line(axes[1, 0], pos_r2, 'Latent Position R² (Higher is better)', 'R²', ylim=(-0.05, 1.05))

    # 4. Latent Velocity R2
    plot_line(axes[1, 1], vel_r2, 'Latent Velocity R² (Higher is better)', 'R²', ylim=(-0.05, 1.05))

    folder_name = os.path.basename(os.path.dirname(os.path.abspath(json_path)))
    plt.suptitle(f"Evaluation Trend: {folder_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_png = os.path.splitext(json_path)[0] + '.png'
    plt.savefig(out_png, dpi=300)
    print(f"Line plot saved to: {out_png}")

if __name__ == '__main__':
    main()
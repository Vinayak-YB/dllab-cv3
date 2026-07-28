import os
import argparse
# Import the base environment simulator or dataset generator class
# from simulator import PhysicsSimulator

def generate_ood_splits(base_data_dir, env_type, n_trajectories=20):
    splits = {
        "ood_gravity": {"gravity_scale": 1.8, "vel_scale": 1.0, "spawn_region": "full"},
        "ood_velocity": {"gravity_scale": 1.0, "vel_scale": 2.2, "spawn_region": "full"},
        "ood_position": {"gravity_scale": 1.0, "vel_scale": 1.0, "spawn_region": "corners_only"},
    }
    
    for split_name, config in splits.items():
        output_dir = f"{base_data_dir}-{env_type}-{split_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating {n_trajectories} trajectories for {split_name} in {output_dir}...")
        
        # Parameters into your physics simulation loop
        # simulator = PhysicsSimulator(
        #     env_type=env_type,
        #     gravity_scale=config["gravity_scale"],
        #     vel_scale=config["vel_scale"],
        #     spawn_region=config["spawn_region"]
        # )
        # simulator.generate_and_save(output_dir, n_trajectories)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["billiard", "magnetic_wells"], required=True)
    parser.add_argument("--n_trajectories", type=int, default=20)
    args = parser.parse_args()
    
    generate_ood_splits("physics-data", args.env, args.n_trajectories)
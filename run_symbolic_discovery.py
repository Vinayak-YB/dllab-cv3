import argparse
import os
import numpy as np

from pysr import PySRRegressor
import torch

from eval_latent_flow import load_model, get_device, build_rollout_loader, resolve_eval_dirs

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def extract_rollout_kinematics(model, dataloader, device, rollout_steps=10, fm_steps=20, max_trajs=100):
    model.eval()
    all_states = []
    n_trajs = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                input_seq, _, _, _ = batch
            else:
                input_seq, _ = batch

            input_seq = input_seq.to(device)
            current_context = input_seq.clone()
            batch_states = []

            for _ in range(rollout_steps):
                rep = model.get_context_representation(current_context)
                state_pred = model.state_head(rep["context"])
                batch_states.append(state_pred.detach().cpu().numpy())

                pred_next = model.predict_next_frame(current_context, num_steps=fm_steps)
                current_context = torch.cat([current_context[:, 1:], pred_next.unsqueeze(1)], dim=1)

            batch_states = np.stack(batch_states, axis=1)
            all_states.append(batch_states)

            n_trajs += batch_states.shape[0]
            if n_trajs >= max_trajs:
                break

    states = np.concatenate(all_states, axis=0)
    return states[:max_trajs]


def build_regression_dataset(predicted_states, target="ax", max_samples=None, filter_bounces=True, seed=42):
    rng = np.random.default_rng(seed)

    X_data = []
    y_data = []

    B, T, D = predicted_states.shape
    if D < 4:
        raise ValueError(f"Expected at least 4 states [x, y, vx, vy], found D={D}")

    for b in range(B):
        for t in range(T - 1):
            x, y, vx, vy = predicted_states[b, t][:4]
            x_next, y_next, vx_next, vy_next = predicted_states[b, t + 1][:4]

            ax = vx_next - vx
            ay = vy_next - vy

            X_data.append([x, y, vx, vy])

            if target == "ax":
                y_data.append(ax)
            elif target == "ay":
                y_data.append(ay)
            else:
                raise ValueError("Target must be 'ax' or 'ay'")

    X = np.asarray(X_data, dtype=np.float32)
    y = np.asarray(y_data, dtype=np.float32)

    if filter_bounces and len(y) > 0:
        p_low = np.percentile(y, 5)
        p_high = np.percentile(y, 95)
        mask = (y >= p_low) & (y <= p_high)
        X = X[mask]
        y = y[mask]
        print(f"Filtered bounces: kept {len(y)} samples within bounds [{p_low:.3f}, {p_high:.3f}]")

    if max_samples is not None and len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y


def main(args):
    device = get_device()
    print(f"Loading model from: {args.model_dir}/{args.ckpt_name}")

    model, ckpt_args = load_model(args.model_dir, args.ckpt_name, device)
    context = ckpt_args.get("context", 5)
    grayscale = ckpt_args.get("grayscale", True)
    invert = ckpt_args.get("invert", False)

    _, test_dirs, _ = resolve_eval_dirs(args)

    dataloader = build_rollout_loader(
        sequence_dirs=test_dirs,
        context=context,
        grayscale=grayscale,
        invert=invert,
        rollout_steps=args.rollout_steps,
        num_workers=0,
        batch_size=args.batch_size,
    )

    print(f"Generating rollout ({args.rollout_steps} steps) and abstracting physical states...")
    predicted_states = extract_rollout_kinematics(
        model=model,
        dataloader=dataloader,
        device=device,
        rollout_steps=args.rollout_steps,
        fm_steps=args.fm_steps,
        max_trajs=args.max_trajs,
    )

    X, y = build_regression_dataset(
        predicted_states,
        target=args.target,
        max_samples=args.max_samples,
        filter_bounces=args.filter_bounces,
        seed=args.seed,
    )

    print(f"Starting PySR for symbolic discovery of '{args.target}' on {len(X)} samples...")

    pysr_model = PySRRegressor(
        niterations=args.pysr_iters,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt"],
        model_selection="best",
        timeout_in_seconds=args.timeout,
        progress=False,
        verbosity=0,
        print_precision=5,
        maxsize=args.maxsize,
    )

    pysr_model.fit(X, y, variable_names=["x", "y", "vx", "vy"])

    print(f"\nLearned equation for {args.target.upper()}:")
    print(pysr_model.sympy())
    print("\n")

    out_csv = os.path.join(args.model_dir, f"pysr_equations_{args.target}.csv")
    pysr_model.equations_.to_csv(out_csv, index=False)
    print(f"Equations saved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Symbolic Discovery on Latent Physics")

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best.pt")

    parser.add_argument("--probe_train_dir", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rollout_steps", type=int, default=15)
    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_trajs", type=int, default=100)

    parser.add_argument("--target", type=str, choices=["ax", "ay"], default="ay")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--filter_bounces", action="store_true", help="Filter out collision impulse spikes")

    parser.add_argument("--pysr_iters", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--maxsize", type=int, default=20)

    args = parser.parse_args()
    main(args)

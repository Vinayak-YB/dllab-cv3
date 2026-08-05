# Evaluation results

The project contains two complementary evaluation suites.

## Cross-environment evaluation

File: `env_summary.md`

Compares ConvLSTM, ConvGRU, and Latent Flow on:
- baseline bouncing ball;
- magnetic wells;
- billiard environment.

These runs evaluate model performance across different physical systems.

### Gravity generalization

**File:** `gravity_generalization_summary.md`

This is the main table for the controlled gravity experiment.

It compares:

- ConvGRU;
- StateMLP;
- Latent Flow.

Each model is trained with either gravity `-1` or gravity `-3` and evaluated
on the same shared test trajectories at both gravity values.

The table therefore covers the complete transfer matrix:

- train `-1` → test `-1` (ID);
- train `-1` → test `-3` (OOD gravity);
- train `-3` → test `-3` (ID);
- train `-3` → test `-1` (OOD gravity).

The primary comparable metrics are:

- one-step position AEE;
- 10-step rollout position AEE.

# Evaluation results

The project contains two complementary evaluation suites.

## Cross-environment evaluation

File: `env_summary.md`

Compares LSTM, GRU, and Latent Flow on:
- baseline bouncing ball;
- magnetic wells;
- billiard environment.

These runs evaluate model performance across different physical systems.

## Gravity and OOD evaluation

File: `g3_evaluation_summary.md`

Compares recurrent, StateMLP, and Latent Flow models on:
- ID test data;
- OOD gravity;
- OOD velocity;
- OOD initial position.

These runs evaluate generalization under changes to physical parameters.

The Latent Flow model appears in both suites and connects the two sets of
experiments. Results from different suites should not be directly ranked
unless training data and evaluation conditions are equivalent.

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

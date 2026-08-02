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
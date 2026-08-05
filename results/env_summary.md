# Environment evaluation summary

Completed models were retrained on environment-specific shared train/validation splits and evaluated on the same held-out trajectories within each environment.

Common observed-space protocol: context `5`, rollout horizon `10`, evaluation stride `5`, and the same robust bright-foreground extractor.

Only position AEE is used for direct model comparison. Lower is better.

| Environment | Model | Run | 1-step Pos AEE (↓) | 1-step failures | Rollout Pos AEE (↓) | Rollout failures |
|---|---|---|---:|---:|---:|---:|
| baseline | lstm | envv2_baseline_lstm | 0.6646 | 0 / 950 | 4.5416 | 0 / 9000 |
| baseline | gru | envv2_baseline_gru | 0.5877 | 0 / 950 | 2.2904 | 0 / 9000 |
| baseline | latent_flow | envv2_baseline_latent_flow | 0.8339 | 0 / 950 | 3.9140 | 0 / 9000 |
| magnetic_wells | lstm | envv2_magwells_lstm | 1.0590 | 0 / 950 | 32.1373 | 0 / 9000 |
| magnetic_wells | gru | envv2_magwells_gru | 0.9621 | 0 / 950 | 16.4702 | 0 / 9000 |
| magnetic_wells | latent_flow | envv2_magwells_latent_flow | 1.1497 | 0 / 950 | 11.8786 | 0 / 9000 |
| billiard | lstm | envv2_billiard_lstm | 0.7535 | 0 / 950 | 8.0461 | 0 / 9000 |
| billiard | gru | envv2_billiard_gru | — | — | — | — |
| billiard | latent_flow | envv2_billiard_latent_flow | 1.1203 | 0 / 950 | 5.5878 | 0 / 9000 |

Latent Flow uses auxiliary ground-truth state supervision during training; LSTM and GRU are trained from image reconstruction only.

The billiard dataset used for this table must be the regenerated version with randomized, non-overlapping initial positions and randomized velocities.

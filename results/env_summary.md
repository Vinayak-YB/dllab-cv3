# Environment evaluation summary

Comparable observed-space AEE for recurrent baselines and latent flow.

| Environment | Model | Run | 1-step Pos AEE | 1-step Vel AEE | Rollout Pos AEE | Rollout Vel AEE |
|---|---|---|---:|---:|---:|---:|
| baseline | lstm | lstm_1 | 1.0401 | 0.7846 | 35.0225 | 18.9535 |
| baseline | gru | gru_0 | 0.6607 | 0.5906 | 30.7676 | 10.3326 |
| baseline | latent_flow | latent_flow_baseline_genloss_e30 | 1.0702 | 1.0702 | 11.3476 | 3.5251 |
| magnetic_wells | lstm | lstm_2 | 2.9564 | 3.7844 | 47.2991 | 34.3512 |
| magnetic_wells | gru | gru_2 | 2.9733 | 3.8049 | 38.5919 | 27.2916 |
| magnetic_wells | latent_flow | latent_flow_magwells_genloss_e30 | 1.7965 | 1.7965 | 13.7267 | 3.7227 |
| billiard | lstm | lstm_3 | 28.4559 | 31.6559 | 57.0236 | 49.5022 |
| billiard | gru | gru_1 | 12.7175 | 19.5697 | 46.5944 | 34.4235 |
| billiard | latent_flow | latent_flow_billiard_genloss_e100_gen03_steps8_state02 | 0.0268 | 0.0268 | 0.0505 | 0.0714 |

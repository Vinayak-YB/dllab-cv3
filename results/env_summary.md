# Environment evaluation summary

Comparable observed-space AEE for recurrent baselines and latent flow.

| Environment | Model | Run | 1-step Pos AEE | 1-step Vel AEE | Rollout Pos AEE | Rollout Vel AEE |
|---|---|---|---:|---:|---:|---:|
| baseline | lstm | lstm_1 | 3.9421 | 2.5013 | 52.3170 | 10.6800 |
| baseline | gru | gru_0 | 6.1365 | 5.4539 | 55.0748 | 17.2301 |
| baseline | latent_flow | latent_flow_baseline_genloss_e30 | 1.0702 | 1.0702 | 11.3476 | 3.5251 |
| magnetic_wells | lstm | lstm_2 | 5.4812 | 3.1516 | 66.5811 | 36.6104 |
| magnetic_wells | gru | gru_2 | 5.1670 | 3.0308 | 54.9117 | 42.4505 |
| magnetic_wells | latent_flow | latent_flow_magwells_genloss_e30 | 1.7965 | 1.7965 | 13.7267 | 3.7227 |
| billiard | lstm | lstm_3 | 3.9912 | 19.2445 | 46.8433 | 50.0258 |
| billiard | gru | gru_1 | 4.8502 | 25.0352 | 38.3231 | 33.0186 |
| billiard | latent_flow | latent_flow_billiard_genloss_e100_gen03_steps8_state02 | 0.0268 | 0.0268 | 0.0505 | 0.0714 |



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


The recurrent baseline results and latent-flow results were produced with different training/evaluation scripts.

## Latent-flow training details

Latent flow was trained with generated-frame supervision. This was important because the first billiard run without 
stronger generated-frame supervision produced noisy extra blobs. The final billiard run used stronger supervision and 
produced clean qualitative rollouts with both balls preserved.

Main latent-flow runs:

```text
baseline:
  run: latent_flow_baseline_genloss_e30
  data_dir: physics-data-v3
  epochs: 30
  generated_frame_loss_weight: 0.2
  generation_loss_steps: 5

magnetic_wells:
  run: latent_flow_magwells_genloss_e30
  data_dir: physics-data-magnetic_wells
  epochs: 30
  generated_frame_loss_weight: 0.2
  generation_loss_steps: 5

billiard:
  run: latent_flow_billiard_genloss_e100_gen03_steps8_state02
  data_dir: physics-data-billiard
  epochs: 100
  generated_frame_loss_weight: 0.3
  generation_loss_steps: 8
  state_loss_weight: 0.2
  num_objects: 2
# Evaluation summary

Comparable metrics across models use observed-space AEE for one-step and rollout.

| Model | Run | Split | 1-step Pos AEE | 1-step Vel AEE | Rollout Pos AEE | Rollout Vel AEE |
|---|---|---|---:|---:|---:|---:|
| recurrent | recurrent_g3 | test_id | 0.6320 | 0.8378 | 9.6605 | 3.4940 |
| recurrent | recurrent_g3 | test_ood_gravity | 2.4343 | 1.4436 | 13.0138 | 5.7717 |
| recurrent | recurrent_g3 | test_ood_velocity | 0.5543 | 0.7372 | 10.0585 | 3.5364 |
| recurrent | recurrent_g3 | test_ood_position | 0.5982 | 0.7433 | 10.9082 | 3.4189 |
| recurrent | recurrent_id_v2 | test_id | 0.5936 | 0.8384 | 9.6401 | 3.6896 |
| recurrent | recurrent_id_v2 | test_ood_gravity | 1.4365 | 1.3052 | 17.4649 | 6.1257 |
| recurrent | recurrent_id_v2 | test_ood_velocity | 0.8745 | 1.1242 | 12.5973 | 4.7327 |
| recurrent | recurrent_id_v2 | test_ood_position | 0.6333 | 0.8961 | 10.7040 | 4.0431 |
| state_mlp | state_mlp_g3 | test_id | 0.5248 | 0.7286 | 4.1376 | 1.9781 |
| state_mlp | state_mlp_g3 | test_ood_gravity | 1.3970 | 1.0612 | 8.4353 | 3.1363 |
| state_mlp | state_mlp_g3 | test_ood_velocity | 0.4998 | 0.8636 | 4.0296 | 2.1752 |
| state_mlp | state_mlp_g3 | test_ood_position | 0.5275 | 0.7630 | 4.4575 | 2.3161 |
| state_mlp | state_mlp_v2 | test_id | 0.3659 | 0.6057 | 5.3712 | 2.0064 |
| state_mlp | state_mlp_v2 | test_ood_gravity | 1.1708 | 1.1491 | 9.7108 | 3.6873 |
| state_mlp | state_mlp_v2 | test_ood_velocity | 0.4823 | 0.8000 | 6.5334 | 2.3406 |
| state_mlp | state_mlp_v2 | test_ood_position | 0.4089 | 0.6527 | 5.6974 | 2.1563 |
| latent_flow | latent_flow_g3 | test_id | 62.6984 | 62.6984 | 64.0180 | 39.0679 |
| latent_flow | latent_flow_g3 | test_ood_gravity | 60.0579 | 60.0579 | 60.0035 | 39.0210 |
| latent_flow | latent_flow_g3 | test_ood_velocity | 65.1891 | 65.1891 | 64.7193 | 39.1979 |
| latent_flow | latent_flow_g3 | test_ood_position | 68.1035 | 68.1035 | 67.8094 | 39.0011 |
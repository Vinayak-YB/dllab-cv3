# Evaluation summary

Comparable metrics across models use observed-space AEE for one-step and rollout.

| Model | Run | Split | 1-step Pos AEE | 1-step Vel AEE | Rollout Pos AEE | Rollout Vel AEE |
|---|---|---|---:|---:|---:|---:|
| recurrent | recurrent_id_v2 | test_id | 0.5936 | 0.8384 | 9.6401 | 3.6896 |
| recurrent | recurrent_id_v2 | test_ood_gravity | 1.4365 | 1.3052 | 17.4649 | 6.1257 |
| recurrent | recurrent_id_v2 | test_ood_velocity | 0.8745 | 1.1242 | 12.5973 | 4.7327 |
| recurrent | recurrent_id_v2 | test_ood_position | 0.6333 | 0.8961 | 10.7040 | 4.0431 |
| state_mlp | state_mlp_v2 | test_id | 0.3659 | 0.6057 | 5.3712 | 2.0064 |
| state_mlp | state_mlp_v2 | test_ood_gravity | 1.1708 | 1.1491 | 9.7108 | 3.6873 |
| state_mlp | state_mlp_v2 | test_ood_velocity | 0.4823 | 0.8000 | 6.5334 | 2.3406 |
| state_mlp | state_mlp_v2 | test_ood_position | 0.4089 | 0.6527 | 5.6974 | 2.1563 |
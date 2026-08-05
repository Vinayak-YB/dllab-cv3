# Controlled gravity generalization

All models are evaluated on the same shared test sets: gravity -1 and gravity -3.

| Model | Run | Train g | Domain | Test g | 1-step Pos AEE | Rollout Pos AEE |
|---|---|---:|---|---:|---:|---:|
| recurrent | recurrent_neg1 | -1.0 | ID | -1.0 | 0.4928 | 7.0888 |
| recurrent | recurrent_neg1 | -1.0 | OOD gravity | -3.0 | 1.9278 | 18.8563 |
| recurrent | recurrent_neg3 | -3.0 | OOD gravity | -1.0 | 2.3977 | 16.2074 |
| recurrent | recurrent_neg3 | -3.0 | ID | -3.0 | 0.6296 | 13.9512 |
| state_mlp | state_mlp_neg1 | -1.0 | ID | -1.0 | 0.4001 | 4.1284 |
| state_mlp | state_mlp_neg1 | -1.0 | OOD gravity | -3.0 | 1.7240 | 9.2072 |
| state_mlp | state_mlp_neg3 | -3.0 | OOD gravity | -1.0 | 1.4536 | 8.5500 |
| state_mlp | state_mlp_neg3 | -3.0 | ID | -3.0 | 0.5074 | 4.2780 |
| latent_flow | latent_flow_neg1_genloss | -1.0 | ID | -1.0 | 1.1107 | 12.2736 |
| latent_flow | latent_flow_neg1_genloss | -1.0 | OOD gravity | -3.0 | 2.3506 | 20.3997 |
| latent_flow | latent_flow_neg3_genloss | -3.0 | OOD gravity | -1.0 | 2.0889 | 17.7167 |
| latent_flow | latent_flow_neg3_genloss | -3.0 | ID | -3.0 | 1.1485 | 14.2337 |
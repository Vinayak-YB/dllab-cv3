python generate_dataset.py --env bouncing_ball --output_dir data/env_v2/baseline/train --num_trajectories 300 --num_frames 100 --seed 42 --overwrite

python generate_dataset.py --env bouncing_ball --output_dir data/env_v2/baseline/val --num_trajectories 50 --num_frames 100 --seed 43 --overwrite

python generate_dataset.py --env bouncing_ball --output_dir data/env_v2/baseline/test --num_trajectories 50 --num_frames 100 --seed 44 --overwrite

python generate_dataset.py --env magnetic_wells --output_dir data/env_v2/magnetic_wells/train --num_trajectories 300 --num_frames 100 --seed 142 --overwrite

python generate_dataset.py --env magnetic_wells --output_dir data/env_v2/magnetic_wells/val --num_trajectories 50 --num_frames 100 --seed 143 --overwrite

python generate_dataset.py --env magnetic_wells --output_dir data/env_v2/magnetic_wells/test --num_trajectories 50 --num_frames 100 --seed 144 --overwrite

python generate_dataset.py --env billiard --output_dir data/env_v2/billiard/train --num_trajectories 300 --num_frames 100 --seed 242 --overwrite

python generate_dataset.py --env billiard --output_dir data/env_v2/billiard/val --num_trajectories 50 --num_frames 100 --seed 243 --overwrite

python generate_dataset.py --env billiard --output_dir data/env_v2/billiard/test --num_trajectories 50 --num_frames 100 --seed 244 --overwrite

python train_latent_flow.py --train_dir data/env_v2/baseline/train --val_dir data/env_v2/baseline/val --save_dir checkpoints --run_name envv2_baseline_latent_flow --epochs 30 --batch_size 8 --context 5 --grayscale --recon_loss_weight 0.2 --motion_loss_weight 0.1 --state_loss_weight 0.1 --generated_frame_loss_weight 0.2 --generation_loss_steps 5 --device cuda --num_workers 0 --seed 42

python train_latent_flow.py --train_dir data/env_v2/magnetic_wells/train --val_dir data/env_v2/magnetic_wells/val --save_dir checkpoints --run_name envv2_magwells_latent_flow --epochs 30 --batch_size 8 --context 5 --grayscale --recon_loss_weight 0.2 --motion_loss_weight 0.1 --state_loss_weight 0.1 --generated_frame_loss_weight 0.2 --generation_loss_steps 5 --device cuda --num_workers 0 --seed 42

python train_latent_flow.py --train_dir data/env_v2/billiard/train --val_dir data/env_v2/billiard/val --save_dir checkpoints --run_name envv2_billiard_latent_flow --epochs 30 --batch_size 8 --context 5 --grayscale --recon_loss_weight 0.2 --motion_loss_weight 0.1 --state_loss_weight 0.1 --generated_frame_loss_weight 0.2 --generation_loss_steps 5 --device cuda --num_workers 0 --seed 42

python train.py --train_dir data/env_v2/baseline/train --val_dir data/env_v2/baseline/val --run_name envv2_baseline_lstm --rnn_type lstm --max_epochs 30 --batch_size 16 --context 5 --rollout 1 --num_workers 0 --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python train.py --train_dir data/env_v2/magnetic_wells/train --val_dir data/env_v2/magnetic_wells/val --run_name envv2_magwells_lstm --rnn_type lstm --max_epochs 30 --batch_size 16 --context 5 --rollout 1 --num_workers 0 --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python train.py --train_dir data/env_v2/billiard/train --val_dir data/env_v2/billiard/val --run_name envv2_billiard_lstm --rnn_type lstm --max_epochs 30 --batch_size 16 --context 5 --rollout 1 --num_workers 0 --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python train.py --train_dir data/env_v2/baseline/train --val_dir data/env_v2/baseline/val --run_name envv2_baseline_gru --rnn_type gru --max_epochs 30 --batch_size 16 --context 5 --rollout 1 --num_workers 0 --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python train.py --train_dir data/env_v2/magnetic_wells/train --val_dir data/env_v2/magnetic_wells/val --run_name envv2_magwells_gru --rnn_type gru --max_epochs 30 --batch_size 16 --context 5 --rollout 1 --num_workers 0 --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python train.py --train_dir data/env_v2/billiard/train --val_dir data/env_v2/billiard/val --run_name envv2_billiard_gru --rnn_type gru --max_epochs 30 --batch_size 16 --context 5 --rollout 1 --num_workers 0 --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python eval_env_recurrent.py --model_dir models/envv2_baseline_gru --data_dir data/env_v2/baseline/test --ckpt_name best.ckpt --context 5 --rollout_steps 10 --eval_stride 5 --num_objects 1 --frame_batch_size 64 --rollout_batch_size 16 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python eval_env_recurrent.py --model_dir models/envv2_magwells_gru --data_dir data/env_v2/magnetic_wells/test --ckpt_name best.ckpt --context 5 --rollout_steps 10 --eval_stride 5 --num_objects 1 --frame_batch_size 64 --rollout_batch_size 16 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python eval_env_recurrent.py --model_dir models/envv2_billiard_gru --data_dir data/env_v2/billiard/test --ckpt_name best.ckpt --context 5 --rollout_steps 10 --eval_stride 5 --num_objects 2 --frame_batch_size 64 --rollout_batch_size 16 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python eval_env_recurrent.py --model_dir models/envv2_baseline_lstm --data_dir data/env_v2/baseline/test --ckpt_name best.ckpt --context 5 --rollout_steps 10 --eval_stride 5 --num_objects 1 --frame_batch_size 64 --rollout_batch_size 16 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python eval_env_recurrent.py --model_dir models/envv2_magwells_lstm --data_dir data/env_v2/magnetic_wells/test --ckpt_name best.ckpt --context 5 --rollout_steps 10 --eval_stride 5 --num_objects 1 --frame_batch_size 64 --rollout_batch_size 16 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python eval_env_recurrent.py --model_dir models/envv2_billiard_lstm --data_dir data/env_v2/billiard/test --ckpt_name best.ckpt --context 5 --rollout_steps 10 --eval_stride 5 --num_objects 2 --frame_batch_size 64 --rollout_batch_size 16 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python eval_latent_flow.py --model_dir checkpoints/envv2_baseline_latent_flow --data_dir data/env_v2/baseline/test --probe_train_dir data/env_v2/baseline/train --ckpt_name best.pt --context 5 --rollout_steps 10 --fm_steps 20 --eval_stride 5 --num_objects 1 --frame_batch_size 16 --rollout_batch_size 8 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python eval_latent_flow.py --model_dir checkpoints/envv2_magwells_latent_flow --data_dir data/env_v2/magnetic_wells/test --probe_train_dir data/env_v2/magnetic_wells/train --ckpt_name best.pt --context 5 --rollout_steps 10 --fm_steps 20 --eval_stride 5 --num_objects 1 --frame_batch_size 16 --rollout_batch_size 8 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python eval_latent_flow.py --model_dir checkpoints/envv2_billiard_latent_flow --data_dir data/env_v2/billiard/test --probe_train_dir data/env_v2/billiard/train --ckpt_name best.pt --context 5 --rollout_steps 10 --fm_steps 20 --eval_stride 5 --num_objects 2 --frame_batch_size 16 --rollout_batch_size 8 --threshold 0.5 --min_mass 3 --fallback_thresholds 0.4 0.3 0.2 0.15 0.1 0.05 --topk_ratio 0.01 --num_workers 0 --seed 42

python build_main_env_results_table.py --root_dir . --out_dir results

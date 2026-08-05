python generate_bouncing_ball_data.py \
  --data_root data \
  --dataset_name bouncing_ball_neg1 \
  --id_gravity -1.0 \
  --ood_gravity -3.0 \
  --id_velocity_scale 5.0 \
  --ood_velocity_scale 9.0 \
  --n_train 500 \
  --n_val 100 \
  --n_test_id 100 \
  --n_test_ood 100 \
  --seed 42 \
  --overwrite_id \
  --overwrite_ood



python generate_bouncing_ball_data.py --data_root data --dataset_name bouncing_ball_neg3 --id_gravity -3.0 --ood_gravity -1.0 --id_velocity_scale 5.0 --ood_velocity_scale 9.0 --n_train 500 --n_val 100 --n_test_id 100 --n_test_ood 100 --seed 42 --overwrite_id --overwrite_ood

python train_recurrent_model.py --train_dir data/bouncing_ball_neg1/train --val_dir data/bouncing_ball_neg1/val --save_dir checkpoints --run_name recurrent_neg1 --epochs 30 --batch_size 16 --context 5 --loss_type weighted_mse --fg_weight 15.0 --fg_threshold 0.5 --num_workers 0 --seed 42


python train_recurrent_model.py --train_dir data/bouncing_ball_neg3/train --val_dir data/bouncing_ball_neg3/val --save_dir checkpoints --run_name recurrent_neg3 --epochs 30 --batch_size 16 --context 5 --loss_type weighted_mse --fg_weight 15.0 --fg_threshold 0.5 --num_workers 0 --seed 42

python train_state_model.py --train_dir data/bouncing_ball_neg1/train --val_dir data/bouncing_ball_neg1/val --save_dir checkpoints --run_name state_mlp_neg1 --epochs 30 --batch_size 256 --context 5 --num_workers 0 --seed 42


python train_state_model.py --train_dir data/bouncing_ball_neg3/train --val_dir data/bouncing_ball_neg3/val --save_dir checkpoints --run_name state_mlp_neg3 --epochs 30 --batch_size 256 --context 5 --num_workers 0 --seed 42

python train_latent_flow.py --train_dir data/bouncing_ball_neg1/train --val_dir data/bouncing_ball_neg1/val --save_dir checkpoints --run_name latent_flow_neg1_genloss --epochs 30 --batch_size 8 --context 5 --grayscale --recon_loss_weight 0.2 --motion_loss_weight 0.1 --state_loss_weight 0.1 --generated_frame_loss_weight 0.2 --generation_loss_steps 5 --device cuda --num_workers 0 --seed 42

python train_latent_flow.py --train_dir data/bouncing_ball_neg3/train --val_dir data/bouncing_ball_neg3/val --save_dir checkpoints --run_name latent_flow_neg3_genloss --epochs 30 --batch_size 8 --context 5 --grayscale --recon_loss_weight 0.2 --motion_loss_weight 0.1 --state_loss_weight 0.1 --generated_frame_loss_weight 0.2 --generation_loss_steps 5 --device cuda --num_workers 0 --seed 42


python eval_recurrent_model.py --model_dir checkpoints/recurrent_neg1 --data_dir data/bouncing_ball_neg1/test_id --probe_train_dir data/bouncing_ball_neg1/train --ckpt_name best.pt --context 5 --rollout_steps 10 --threshold 0.5 --num_workers 0 --seed 42

python eval_recurrent_model.py --model_dir checkpoints/recurrent_neg1 --data_dir data/bouncing_ball_neg1/test_ood_gravity --probe_train_dir data/bouncing_ball_neg1/train --ckpt_name best.pt --context 5 --rollout_steps 10 --threshold 0.5 --num_workers 0 --seed 42

python eval_recurrent_model.py --model_dir checkpoints/recurrent_neg3 --data_dir data/bouncing_ball_neg1/test_id --probe_train_dir data/bouncing_ball_neg3/train --ckpt_name best.pt --context 5 --rollout_steps 10 --threshold 0.5 --num_workers 0 --seed 42

python eval_recurrent_model.py --model_dir checkpoints/recurrent_neg3 --data_dir data/bouncing_ball_neg1/test_ood_gravity --probe_train_dir data/bouncing_ball_neg3/train --ckpt_name best.pt --context 5 --rollout_steps 10 --threshold 0.5 --num_workers 0 --seed 42


python eval_state.py --model_dir checkpoints/state_mlp_neg1 --data_dir data/bouncing_ball_neg1/test_id --probe_train_dir data/bouncing_ball_neg1/train --ckpt_name best.pt --context 5 --rollout_steps 10 --num_workers 0 --seed 42

python eval_state.py --model_dir checkpoints/state_mlp_neg1 --data_dir data/bouncing_ball_neg1/test_ood_gravity --probe_train_dir data/bouncing_ball_neg1/train --ckpt_name best.pt --context 5 --rollout_steps 10 --num_workers 0 --seed 42

python eval_state.py --model_dir checkpoints/state_mlp_neg3 --data_dir data/bouncing_ball_neg1/test_id --probe_train_dir data/bouncing_ball_neg3/train --ckpt_name best.pt --context 5 --rollout_steps 10 --num_workers 0 --seed 42

python eval_state.py --model_dir checkpoints/state_mlp_neg3 --data_dir data/bouncing_ball_neg1/test_ood_gravity --probe_train_dir data/bouncing_ball_neg3/train --ckpt_name best.pt --context 5 --rollout_steps 10 --num_workers 0 --seed 42


python eval_latent_flow.py --model_dir checkpoints/latent_flow_neg1_genloss --data_dir data/bouncing_ball_neg1/test_id --probe_train_dir data/bouncing_ball_neg1/train --ckpt_name best.pt --context 5 --rollout_steps 10 --fm_steps 20 --eval_stride 1 --num_objects 1 --frame_batch_size 16 --rollout_batch_size 8 --num_workers 0 --seed 42

python eval_latent_flow.py --model_dir checkpoints/latent_flow_neg1_genloss --data_dir data/bouncing_ball_neg1/test_ood_gravity --probe_train_dir data/bouncing_ball_neg1/train --ckpt_name best.pt --context 5 --rollout_steps 10 --fm_steps 20 --eval_stride 1 --num_objects 1 --frame_batch_size 16 --rollout_batch_size 8 --num_workers 0 --seed 42

python eval_latent_flow.py --model_dir checkpoints/latent_flow_neg3_genloss --data_dir data/bouncing_ball_neg1/test_id --probe_train_dir data/bouncing_ball_neg3/train --ckpt_name best.pt --context 5 --rollout_steps 10 --fm_steps 20 --eval_stride 1 --num_objects 1 --frame_batch_size 16 --rollout_batch_size 8 --num_workers 0 --seed 42

python eval_latent_flow.py --model_dir checkpoints/latent_flow_neg3_genloss --data_dir data/bouncing_ball_neg1/test_ood_gravity --probe_train_dir data/bouncing_ball_neg3/train --ckpt_name best.pt --context 5 --rollout_steps 10 --fm_steps 20 --eval_stride 1 --num_objects 1 --frame_batch_size 16 --rollout_batch_size 8 --num_workers 0 --seed 42

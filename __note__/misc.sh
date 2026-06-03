#: norm state
# Norm stats not found in /data0/luokang/research/openpi/openpi/assets/pi05_libero/<repo_id>
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
JAX_PLATFORMS=cpu \
python scripts/compute_norm_stats.py \
--config-name pi05_libero

HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
JAX_PLATFORMS=cpu \
python scripts/compute_norm_stats.py \
--config-name pi05_libero_custom

#: convert
python examples/convert_jax_model_to_pytorch.py \
--checkpoint_dir ckpts/openpi-assets/checkpoints/pi05_base \
--config_name pi05_droid \
--output_path ckpts/openpi-assets/checkpoints_torch/pi05_base

#: wandb
wandb sync /data0/luokang/research/openpi/openpi/wandb/run-20260602_063408-6p97bdbe
wandb sync /data0/luokang/research/openpi/openpi/wandb/run-20260602_155640-uirhmejs
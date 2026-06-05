# CKPT_DIR=checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06020408/30000
# CKPT_DIR=ckpts/openpi-assets/checkpoints/pi05_libero
# CKPT_DIR=checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06011512/30000
CKPT_DIR=checkpoints/pi0_ar_thinking_libero_custom_low_mem_finetune/pi0_ar_thinking_libero_custom_low_mem_finetune-06040441/25000


HF_HOME=/data0/luokang/.cache/huggingface \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
python scripts/serve_policy.py \
--port 8003 \
policy:checkpoint \
--policy.config=pi0_ar_thinking_libero_custom_low_mem_finetune \
--policy.dir $CKPT_DIR

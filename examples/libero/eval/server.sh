# CKPT_DIR=checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06020408/15000
# CKPT_DIR=checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06011512/15000
CKPT_DIR=checkpoints/pi05_libero_custom_low_mem_finetune/pi05_libero_custom_low_mem_finetune-06020633/17000

HF_HOME=/data0/luokang/.cache/huggingface \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=6 \
python scripts/serve_policy.py policy:checkpoint \
--policy.config=pi05_libero_low_mem_finetune \
--policy.dir $CKPT_DIR


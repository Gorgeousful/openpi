# 实测单卡16，显存占用为47G 注意这里是mix: weight,gradient fp32; activation,computation bf16
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
WANDB_MODE=disabled \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
CUDA_VISIBLE_DEVICES=4,5 \
python scripts/train.py \
pi05_libero \
--exp-name=pi05_libero \
--batch_size 32 \
--gradient_accumulation_steps 1 \
--fsdp-devices 2 \
--ema-decay None \
--overwrite

# 实测单卡32，显存占用约33G XLA_PYTHON_CLIENT_PREALLOCATE=false 
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
WANDB_MODE=disabled \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
CUDA_VISIBLE_DEVICES=3,4 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune \
--batch_size 64 \
--gradient_accumulation_steps 1 \
--ema-decay None \
--overwrite


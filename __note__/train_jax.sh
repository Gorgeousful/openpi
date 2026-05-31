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

HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
WANDB_MODE=disabled \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
CUDA_VISIBLE_DEVICES=4,5 \
python scripts/train.py \
pi05_libero \
--exp-name=pi05_libero \
--batch_size 32 \
--gradient_accumulation_steps 2 \
--fsdp-devices 2 \
--ema-decay None \
--overwrite


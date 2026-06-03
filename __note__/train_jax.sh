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


#TODO debug
import rich; from rich.console import Console; cs = Console()
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
WANDB_MODE=disabled \
JAX_PLATFORMS=cpu \
DEBUG=1 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-test \
--overwrite

#TODO test
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
WANDB_MODE=disabled \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-test \
--batch_size 64 \
--overwrite

# lora微调实测单卡32，显存占用约33G; 单卡64, 显存占用约44G  
# XLA_PYTHON_CLIENT_PREALLOCATE=false   --resume
#: 1k warmup + 2.5e-5 lr + discrete state
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,4 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-06011512 \
--resume

#: 1k warmup + 2.5e-5 lr + speical discrete state
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=2,3 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=2,3 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-06020408 \
--resume

#: 1k warmup + 2.5e-5 lr + special discrete state + aux(include fast) + detach
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,4 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-06020633 \
--resume


# ==========================================================================================

#: 1k warmup + 2.5e-5 lr + discrete state + aux(include fast) + detach
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=2,3 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite
#: 1k warmup + 2.5e-5 lr + discrete state + aux(include fast) + detach + raw pretrain weight
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=0,6 \
timeout -k 1m 6h \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite
#: 10k warmup + 5e-5 lr
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=2,3 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite
#: 5k warmup + 5e-5 lr
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,4 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite
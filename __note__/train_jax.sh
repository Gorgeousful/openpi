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

# ==========================================================================================
# lora微调实测单卡32，显存占用约33G; 单卡64, 显存占用约44G  
# XLA_PYTHON_CLIENT_PREALLOCATE=false   --resume

#: discrete state
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

#: speical discrete state
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=2,3 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=2,3 \
python scripts/train.py \
pi05_libero_low_mem_finetune \
--exp-name=pi05_libero_low_mem_finetune-06020408 \
--resume

#: special discrete state + aux(include fast) + detach
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,4 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-06020633 \
--resume

#: special discrete state + aux(include fast) + detach + 0.1 weight
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

#: special discrete state + aux(include fast) + 0.05 weight
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

# ==========================================================================================

#: fast thinking + discrete state
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python scripts/train.py \
pi0_fast_thinking_libero_custom_low_mem_finetune \
--exp-name=pi0_fast_thinking_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

#: ar thinking + discrete state
# WANDB_MODE=disabled \
# HF_HOME=/data0/luokang/.cache/huggingface \
# HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
# CUDA_VISIBLE_DEVICES=5 \
# python scripts/train.py \
# pi0_ar_thinking_libero_custom_low_mem_finetune \
# --exp-name=pi0_ar_thinking_libero_custom_low_mem_finetune-test \
# --batch_size=16 \
# --overwrite

HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,2 \
python scripts/train.py \
pi0_ar_thinking_libero_custom_low_mem_finetune \
--exp-name=pi0_ar_thinking_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite

HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=1,2 \
python scripts/train.py \
pi0_ar_thinking_libero_custom_low_mem_finetune \
--exp-name=pi0_ar_thinking_libero_custom_low_mem_finetune-06040441 \
--resume

#: oft thinking + discrete state
WANDB_MODE=disabled \
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=1 \
python scripts/train.py \
pi0_oft_thinking_libero_custom_low_mem_finetune \
--exp-name=pi0_oft_thinking_libero_custom_low_mem_finetune-test \
--batch_size=16 \
--overwrite

HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/train.py \
pi0_oft_thinking_libero_custom_low_mem_finetune \
--exp-name=pi0_oft_thinking_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite



# ==========================================================================================

#: discrete state + aux(include fast) + detach
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=2,3 \
python scripts/train.py \
pi05_libero_custom_low_mem_finetune \
--exp-name=pi05_libero_custom_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite
#: discrete state + aux(include fast) + detach + raw pretrain weight
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
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/train.py \
pi05_libero_gram_low_mem_finetune \
--exp-name=pi05_libero_gram_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite


#: gram
WANDB_MODE=disabled \
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/train.py \
pi05_libero_gram_low_mem_finetune \
--exp-name=pi05_libero_gram_low_mem_finetune-$(date +%m%d%H%M) \
--overwrite
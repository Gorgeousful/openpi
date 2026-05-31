# 调试
python scripts/train_pytorch.py \
pi05_libero \
--exp_name my_experiment_pytorch \
--batch_size 4 \
--overwrite

# 实测单卡32，显存占用为42G 注意这是full bf16
HF_HOME=/data0/luokang/.cache/huggingface \
HF_LEROBOT_HOME=/data0/luokang/dataset/luokang \
WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=5,6 \
torchrun --standalone --nnodes=1 \
--nproc_per_node=2 \
scripts/train_pytorch.py \
pi05_libero \
--exp_name pi05_libero \
--batch_size 64 \
--gradient_accumulation_steps 1 \
--overwrite
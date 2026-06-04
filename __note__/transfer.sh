mkdir -p /data0/luokang/dataset/luokang/lerobot/
cd /data0/luokang/dataset/luokang && mkdir ckpts docker conda

rsync -avh --progress -e "ssh -p 22" \
/data0/luokang/dataset/luokang/lerobot/libero/libero_all_no_noops_1.0.0_lerobot_10hz \
luokang@6024.irmv.top:/data0/luokang/dataset/luokang/lerobot/libero/

rsync -avh --progress -e "ssh -p 22" \
/data0/luokang/dataset/luokang/docker/cu126.tar \
luokang@6024.irmv.top:/data0/luokang/dataset/luokang/docker/

rsync -avh --progress -e "ssh -p 22" \
/data0/luokang/dataset/luokang/conda/openpi.tar.gz \
luokang@6024.irmv.top:/data0/luokang/dataset/luokang/conda/

rsync -avh --progress -e "ssh -p 22" \
/data0/luokang/research/openpi/openpi/checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06020408/16000 \
luokang@6024.irmv.top:/data0/luokang/research/openpi/openpi/checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06020408

rsync -avh --progress -e "ssh -p 22" \
/data0/luokang/research/openpi/openpi/checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06020408/wandb_id.txt \
luokang@6024.irmv.top:/data0/luokang/research/openpi/openpi/checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06020408

rsync -avh --progress -e "ssh -p 22" \
/data0/luokang/research/openpi/openpi/ckpts/openpi-assets/checkpoints/paligemma-3b-mix-224-jax \
luokang@6024.irmv.top:/data0/luokang/research/openpi/openpi/ckpts/openpi-assets/checkpoints/
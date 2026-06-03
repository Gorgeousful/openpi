kill -9 $(lsof -t -i:8000)
#: libero
CUDA_VISIBLE_DEVICES=2 \
python scripts/serve_policy.py policy:checkpoint \
--policy.config=pi05_libero_low_mem_finetune \
--policy.dir=checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06020408/15000 \
--port 8003

LIBERO_HOME=/data0/luokang/research/LIBERO \
LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero \
PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} \
python examples/libero/main.py \
--args.task_suite_name libero_10 \
--args.host 127.0.0.1 \
> /data0/luokang/research/openpi/openpi/data/libero/eval_libero.log 2>&1
# CKPT_DIR=checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06020408/30000
# CKPT_DIR=checkpoints/pi05_libero_low_mem_finetune/pi05_libero_low_mem_finetune-06011512/30000
CKPT_DIR=checkpoints/pi0_ar_thinking_libero_custom_low_mem_finetune/pi0_ar_thinking_libero_custom_low_mem_finetune-06040441/25000


TASK_SUITE=libero_10
RESULT_DIR="$CKPT_DIR/result/$TASK_SUITE"
if [ -d "$RESULT_DIR" ]; then
    rm -rf "$RESULT_DIR"
fi
mkdir -p "$RESULT_DIR"

LIBERO_HOME=/data0/luokang/research/LIBERO \
LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero \
PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} \
PYTHONUNBUFFERED=1 \
python examples/libero/main.py \
--args.task_suite_name $TASK_SUITE \
--args.invert_gripper \
--args.normalize_gripper \
--args.video_out_path "$RESULT_DIR/videos" \
--args.result_out_path "$RESULT_DIR/result.json" \
--args.port 8003 \
--args.flip_video \
--args.draw_grounding \
2>&1 | tee "$RESULT_DIR/result.log"




# CKPT_DIR=ckpts/openpi-assets/checkpoints/pi05_libero

# TASK_SUITE=libero_10
# RESULT_DIR="$CKPT_DIR/result/$TASK_SUITE"
# if [ -d "$RESULT_DIR" ]; then
#     rm -rf "$RESULT_DIR"
# fi
# mkdir -p "$RESULT_DIR"

# LIBERO_HOME=/data0/luokang/research/LIBERO \
# LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero \
# PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} \
# PYTHONUNBUFFERED=1 \
# python examples/libero/main.py \
# --args.task_suite_name $TASK_SUITE \
# --args.video_out_path "$RESULT_DIR/videos" \
# --args.result_out_path "$RESULT_DIR/result.json" \
# --args.port 8003 \
# 2>&1 | tee "$RESULT_DIR/result.log"

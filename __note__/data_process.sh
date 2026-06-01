LeRobotDataset
  根据 action_horizon=10 和 fps 构造未来 action chunk
---
PromptFromLeRobotTask
  根据 task_index 生成 prompt
---
RepackTransform
  映射数据集字段名称
---
LiberoCustomInputs
  图像转为 H,W,C uint8
  base 与 wrist 固定水平翻转 (根据情况接入)
  补充 right_wrist 零图像
  构建 image_mask
---
Normalize
  根据 norm_stats 对 state 和 actions 做 z-score 归一化
---
InjectDefaultPrompt
  当前已有 prompt，因此基本不做事
ResizeImages
  resize_with_pad 到 224 x 224
TokenizePrompt
  prompt 转 token
PadStatesAndActions
  state 和 actions 最后一维补齐到 32
---
DataLoaderImpl
Observation.from_dict
  图像从 uint8 [0,255] 转为 float32 [-1,1]
---
Forward 内部
JAX:   preprocess_observation
Torch: preprocess_observation_pytorch
  训练时做随机裁剪、旋转和颜色增强
---
模型 forward
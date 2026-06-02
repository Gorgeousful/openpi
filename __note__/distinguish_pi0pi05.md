当前代码里，`pi0` 和 `pi05` 共用同一个 `Pi0` 类，通过 `config.pi05` 分支切换。核心差异主要有三处。

**整体结构**
```text
pi0:
图像 + task prompt
        ↓
VLM backbone
        ↓
连续 state token + MLP([noisy action, timestep])
        ↓
action expert
        ↓
连续 action

pi05:
图像 + task prompt + 离散 state 文本
        ↓
VLM backbone
        ↓
noisy action
        ↓
带 timestep 条件的 adaRMSNorm action expert
        ↓
连续 action
```

## 1. State 注入方式

### Pi0：连续 state token
Pi0 会将归一化后的连续 state 通过线性层映射：

```python
state_token = self.state_proj(obs.state)[:, None, :]
```

然后把它作为 action expert suffix 的第一个 token。

代码：[pi0.py](/data0/luokang/research/openpi/openpi/src/openpi/models/pi0.py:167)

suffix 形式：

```text
[state token] [action token 1] ... [action token H]
```

### Pi05：离散 state 文本
Pi05 不会创建连续 state token。默认情况下，state 会先在 `[-1, 1]` 范围内量化为 256 档，再转成文本放入 VLM prompt：

```python
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
```

代码：[tokenizer.py](/data0/luokang/research/openpi/openpi/src/openpi/models/tokenizer.py:25)

按你当前修改后的格式，prompt 类似：

```text
Task: open the drawer
State: 103 128 42 255 0 91 127
```

### 当前配置需要注意
Pi05 默认：

```python
discrete_state_input = pi05
```

代码：[pi0_config.py](/data0/luokang/research/openpi/openpi/src/openpi/models/pi0_config.py:51)

但是你可以手动覆盖。你目前的两个配置有区别：

```python
pi05_libero_low_mem_finetune:
    discrete_state_input=True
```

```python
pi05_libero_custom_low_mem_finetune:
    discrete_state_input=False
```

第二种情况下，Pi05 既不会把 state 放入 prompt，也不会像 Pi0 一样创建连续 state token。也就是说，**state 完全不会进入模型**。

## 2. Timestep 注入方式

训练 flow matching 时，两者都会使用 timestep `t`，但注入方式不同。

### Pi0：与 action embedding 拼接
Pi0 将 timestep embedding 复制到每个 action token，再与 noisy action 拼接：

```python
[action_embedding, time_embedding]
        ↓
MLP
        ↓
action expert token
```

代码：[pi0.py](/data0/luokang/research/openpi/openpi/src/openpi/models/pi0.py:186)

### Pi05：通过 adaRMSNorm 注入
Pi05 的 noisy action embedding 直接进入 action expert：

```python
action_expert_tokens = action_tokens
```

timestep 单独经过 MLP：

```python
time_emb = self.time_mlp_in(time_emb)
time_emb = self.time_mlp_out(time_emb)
adarms_cond = time_emb
```

代码：[pi0.py](/data0/luokang/research/openpi/openpi/src/openpi/models/pi0.py:178)

action expert 内部每层 RMSNorm 会根据 timestep 生成：

```text
scale
shift
gate
```

因此 timestep 会调制每一层 action expert，而不是仅在输入端与 action 拼接。

## 3. 默认 token 长度

默认配置：

| 模型 | `max_token_len` |
|---|---:|
| Pi0 | 48 |
| Pi05 | 200 |

代码：[pi0_config.py](/data0/luokang/research/openpi/openpi/src/openpi/models/pi0_config.py:47)

Pi05 需要更长 prompt，主要因为离散 state 会变成文本 token。

## 4. 数据归一化

当前数据配置还会自动区分归一化方式：

```python
use_quantile_norm = model_config.model_type != ModelType.PI0
```

代码：[config.py](/data0/luokang/research/openpi/openpi/src/openpi/training/config.py:213)

因此默认行为是：

| 模型 | 归一化 |
|---|---|
| Pi0 | z-score normalization |
| Pi05 | quantile normalization |

## 5. 相同部分

两者都使用相同的：

- PaliGemma VLM backbone；
- 300M action expert；
- flow matching 目标；
- attention 组织方式；
- action chunk 预测；
- 迭代去噪推理；
- KV-cache 前缀复用。

训练目标都仍然是：

```python
x_t = t * noise + (1 - t) * actions
u_t = noise - actions
action_loss = mean((v_t - u_t) ** 2)
```

代码：[pi0.py](/data0/luokang/research/openpi/openpi/src/openpi/models/pi0.py:254)

## 一句话总结

Pi05 不是更换了 VLM 或 action expert，而是重新设计了条件注入：

```text
Pi0  = 连续 state suffix token + 输入端注入 timestep
Pi05 = 离散 state prompt token + 每层 adaRMSNorm 注入 timestep
```

你当前最值得确认的是 `pi05_libero_custom_low_mem_finetune` 中的：

```python
discrete_state_input=False
```

如果不是有意丢弃 state，应该改为 `True`。
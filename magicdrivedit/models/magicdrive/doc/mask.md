# 一次把 “Mask” 弄明白  

下文按 “概念 → 旧实现流程 → 代码落点 → 维度推导 → 改成 6×8 帧后要动什么” 的顺序讲。为了方便对照，下面把出现过的 **4 种 mask** 先列一遍：

| 变量名 | 典型形状（旧） | 作用 | 主要流向 |
| ------ | -------------- | ---- | -------- |
| `mask` | `[B, N_token]` | 文本真实长度 | `encode_text()` |
| `drop_cond_mask` | `[B]` | 一条样本是否做 **完全无条件** | `encode_text() / encode_cam()` |
| `drop_frame_mask` | `[B, T_orig = 8]` | 哪些帧的 **时空条件** 被丢弃 | `encode_box() / encode_cam()` |
| `x_mask` | `[B, T_latent]` | 图像自监督 **随机遮挡** | 进 `forward()` 里的每个 Transformer Block |

下面逐一拆开。

---

## 1. 文本 token mask（`mask`）

### 理论  
一条描述最多 300 token，但实际长度可变。`mask` 告诉模型每个样本真实 token 数，防止把 `<pad>` 位置也算进注意力。

### 代码  
```python
ret = text_encoder.encode(y_txt)
mask = ret["mask"]  # shape [B, 300]
...
y, y_lens = self.encode_text(ret['y'], mask, drop_cond_mask)
```

在 `encode_text()` 里：
```python
y = y.squeeze(1)[:, :max_len]   # 删掉多余 pad
```

### 维度  
* 输入：`[B, 1, N_max, hidden]`
* 输出：`[B, L_real, hidden]`

和 “一次压缩六目” 没半点关系，不用改。

---

## 2. 样本级无条件开关（`drop_cond_mask`）

### 理论  
训练时以概率 `drop_cond_ratio` 把所有条件（文本 + 3D box + 地图 …）清零，让模型学 **纯 U-Net** 能力；推断时就能做 unconditional 生成或者 Classifier-Free Guidance。

### 代码  
1. 在 dataloader 里随机生成  
   ```python
   drop_cond_mask = torch.ones(B)
   if random.random() < drop_cond_ratio:
       drop_cond_mask[bs] = 0
   ```
2. 进入 `encode_text()`  
   ```python
   self.y_embedder(..., force_drop_ids = 1 - drop_cond_mask)
   ```
   里面直接把 token 全换成特殊的 `<null>` embedding。  
3. 进入 `encode_cam()` / `encode_box()` 时用 `repeat(drop_cond_mask, "B -> B T")`  
   影响相机、bbox 这类 token 是否置 0。

### 维度  
始终是一维 `[B]` ，不随 `NC`、`T` 改变——所以 **不用动**。

---

## 3. 帧级 Dropout（`drop_frame_mask`）

### 理论  
• 让模型学会 **缺帧鲁棒**：有时某些帧的盒子 / 运动 / 相机都不可用。  
• 旧实现只有 `T_orig=8` 列，因为每台相机会被拆成 `(B·NC)` 放在 batch 维，“相机差异” 在此之前就消失了。

### 旧流程（Flatten-NC）  
```
pixel_values   [B, T=8, NC=6, ...]
            └─→ x, cams, bbox all reshape → (B·NC, ...)
drop_frame_mask        [B, 8]
repeat(..., NC=NC) ───→ [B·NC, 8]   # 和 batch 对齐
```

### 改为 “6 目 × 8 帧” 后应该这样  
```
pixel_values   [B, T=8, NC=6, ...] ➜  VAE ➜ latent [B, CT=48, ...]
drop_frame_mask 仍然 [B, 8]  (不变!)
encode_* 内部再 repeat 到 48 列，对齐 CT
```

所以 **不要在 dataloader 侧提前 repeat**，由模型里  
```python
repeat(drop_frame_mask, "B T -> B (T NC)", NC=NC)
```  
来完成。

---

## 4. 图像遮挡 mask（`x_mask`）

### 理论  
这是一个可选的 **masked-autoencoder** 目标：  
随机把某些 “micro-frame”（通常与 VAE 下采样对应的 latent-T）设成 0，让模型重建。好处是利用 Diffusion 的 **噪声预测** 同时做时空 inpainting。

### 代码落点  
1. 生成：  
   ```python
   if cfg.mask_ratios:
       mask = mask_generator.get_masks(x)   # [B, T_latent]
   ```
2. `forward()`：  
   * 先把 `timestep` 嵌入拆成 `t_mlp` 和 “空 token” 版本 `t0_mlp`  
   * 在每个 `MultiViewSTDiT3Block` 里：  
     - 若位置被 mask，则把 shift/scale 换成 `t0_mlp` → 相当于 **不让网络看到当前帧内容**。

### 维度  
旧: `[B, T_latent=8]`，因为 NC 已经拼到 batch；  
新: **仍然 `[B, T_latent]`**，不会额外出现 `NC`（它已并入 `T_latent=12`）。

---

## 5. 把四种 mask 放到时间线里

```
                           ┌───────────┐
dataloader                 │ drop_cond │  shape [B]
                           └───────────┘
                                 │
                                 ▼
                         ┌────────────────────┐
                         │ encode_text (y)    │ ← mask [B,N_token]
                         └────────────────────┘
                                 │
                                 ▼
┌───────┐  repeat(. ,NC) ┌───────────────────┐
│ cams  │───────────────→│ encode_cam        │
│ rel   │  repeat(T→CT)  │ encode_frame      │
│ bbox  │───────────────→│ encode_box        │
└───────┘                └───────────────────┘
                                 │
                                 ▼
                       concatenate → cond_tokens
                                 │
                                 ▼
          ┌─ optional ─┐ ┌──────────────────────────┐
          │  x_mask    │→│   STDiT3 Transformer     │
          └────────────┘ └──────────────────────────┘
```

---

## 6. 常见 BUG 一眼能看出的信号

1. **NC 推断成 1**  
   - 现象：`cams.shape[1]` 是 48，但 `repeat(drop_frame_mask, "B (T) -> ...", NC=1)` 不出错；  
   - 结果：后面 reshape `(B, CT, S)` 和 `(B, T, ... )` 对不上；  
   - 定位：打印 `NC`, `drop_frame_mask.shape`, `cams.shape`.

2. **mask.sum() 为 0 或 >sequence_len**  
   - 很多 embedder 在内部会 `assert mask.dim == ...` 或做 `masked_fill`;  
   - 不一致时立即能捕获。

---

## 7. 你需要改动的地方（再次核对）

1. **train loop**  
   ```python
   # model_args["drop_frame_mask"] = drop_frame_mask_ct   # ⬅ 删除
   model_args["drop_frame_mask"] = drop_frame_mask        # ⬅ 保留 8 列
   ```

2. **encode_cond_sequence**  
   当前实现已正确 repeat；不再需要其他修改。

3. **其他 mask（x_mask / drop_cond_mask / mask）**  
   与 “六目一次压缩” 无耦合，无需改。

---

### 小结

* “Mask” 本质就是 **告诉模型哪里不可用**；  
* 我们只改了时空维度的组织方式，并没有改 **需要被遮掉的位置** 的语义；  
* 因此除了确保 repeat 发生在正确的一层（模型里而不是 dataloader 里），其余逻辑 **保持原状** 就能正常工作。  

有了这张“思维导图”，再遇到 shape 报错时，只要对照 “谁负责 repeat、目标维度应该是多少” 就能迅速定位。
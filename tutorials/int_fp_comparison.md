# NVFP4 压缩比分析

## 1. NVFP4 两级缩放存储结构

NVFP4 采用两级缩放方案：

- **Global scale**（全局缩放）：1 个 FP8 (E4M3, 8 bits)，由整组共享
- **Block scale**（块缩放）：每 block 1 个 E4M3 (8 bits)
- **FP4 数据**：每元素 4 bits (E2M1, 符号1位 + 指数2位 + 尾数1位)

单个 block 的存储量（bit）：

$$\text{bits\_per\_block} = 4 \times \text{block\_size} + 8 + \frac{8}{N}$$

其中：
- $4 \times \text{block\_size}$：FP4 数据部分
- $8$：本 block 的 E4M3 block scale
- $\frac{8}{N}$：FP8 global scale 被 N 个 block 分摊（$N$ = 该量化组内的 block 总数）

当 $N$ 较大时，global scale 的开销可忽略，简化为：

$$\text{bits\_per\_block} \approx 4 \times \text{block\_size} + 8$$

## 2. 压缩比公式

等效每元素 bit 数：

$$\text{bits\_per\_element} = 4 + \frac{8}{\text{block\_size}} + \frac{8}{\text{block\_size} \times N}$$

相对于 FP32（32 bits/元素）的压缩比：

$$\text{Compression Ratio} = \frac{32}{\text{bits\_per\_element}} = \frac{32}{4 + \frac{8}{\text{block\_size}} + \frac{8}{\text{block\_size} \times N}}$$

**理论极限**（$\text{block\_size} \to \infty$）：

$$\lim_{\text{block\_size} \to \infty} \text{CR} = \frac{32}{4} = 8\times$$

## 3. 不同 block_size 下的压缩比

（假设 $N$ 足够大，忽略 global scale 分摊）

| block_size | FP4 数据 (bits) | Block Scale (bits) | 总计 (bits) | 每元素 bit | 压缩比 | 相对极限 |
|:----------:|:---------------:|:------------------:|:-----------:|:----------:|:------:|:--------:|
| 16         | 64              | 8                  | 72          | 4.500      | 7.11×  | 88.9%    |
| 32         | 128             | 8                  | 136         | 4.250      | 7.53×  | 94.1%    |
| 64         | 256             | 8                  | 264         | 4.125      | 7.76×  | 97.0%    |
| 128        | 512             | 8                  | 520         | 4.063      | 7.88×  | 98.4%    |
| 256        | 1024            | 8                  | 1032        | 4.031      | 7.94×  | 99.2%    |
| 512        | 2048            | 8                  | 2056        | 4.016      | 7.97×  | 99.6%    |
| 768        | 3072            | 8                  | 3080        | 4.010      | 7.98×  | 99.7%    |
| 1536       | 6144            | 8                  | 6152        | 4.005      | 7.99×  | 99.9%    |

> **关键洞察**：从 block_size=16 升级到 256，block scale 开销从 12.5% 降到 0.78%，压缩比从 7.11× 提升到 7.94×。但继续增大 block_size，边际收益递减（256→1536 只多 0.05× 压缩比，精度却损失更多）。

## 4. 权重量化 vs 激活性量化 的 N 值差异

| 量化类型 | 量化组划分 | N（每组的 block 数） |
|---------|-----------|---------------------|
| **权重量化** (`NVFP4Quantization`) | 每个 output channel 为一个组，沿 `in_features` 分块 | $N = \frac{\text{in\_features}}{\text{block\_size}}$ |
| **激活性量化** (`NVFP4ActivationQuantization`) | 每个 token 为一个组，沿 `dim` 轴分块 | $N = \frac{\text{dim}}{\text{block\_size}}$ |

对于 Sana Sprint 0.6B 模型 ($\text{inner\_dim} = 1536$)：

- 权重量化：G = 1536（每个 output channel 独立），若 block_size=256，则 $N = \frac{1536}{256} = 6$ per channel
- 激活性量化：dim=1536，若 block_size=256，则 $N = 6$ per token

## 5. 与其他量化格式的压缩比对比

| 格式 | 每元素 bit | 压缩比 (vs FP32) | 备注 |
|------|:----------:|:----:|------|
| FP32 (baseline) | 32        | 1×   | 原始精度 |
| BF16            | 16        | 2×   | 混合精度训练 |
| FP16            | 16        | 2×   | 推理加速 |
| INT8            | 8         | 4×   | per-tensor/per-channel scale |
| FP8 (E4M3)      | 8         | 4×   | H100 原生支持 |
| **NVFP4**       | **~4.03** | **~7.94×** | block_size=256，两级缩放 |
| INT4            | 4         | 8×   | 无额外 scale 开销（per-channel scale 分摊极小） |
| NF4             | 4         | ~8×  | 4-bit NormalFloat，QLoRA 中使用 |

> NVFP4 相比纯 INT4/NF4 的主要优势是两级缩放提供更好的精度保留，代价是每个 block 多 8 bit 的 E4M3 block scale。

## 6. block_size 选择的权衡

| block_size | 压缩比 | Block 粒度 | 量化精度 | 适用场景 |
|:----------:|:------:|:---------:|:-------:|---------|
| 小 (16~32) | 7.1×~7.5× | 细粒度 | 高精度 | 精度敏感层，小矩阵 |
| 中 (64~256) | 7.8×~7.9× | 平衡 | 中等 | **推荐默认值** |
| 大 (512~1536) | ~8.0× | 粗粒度 | 低精度 | 极限压缩，大 batch |

---

## 7. 高效 FP4 矩阵乘法的量化布局设计

### 7.1 核心原则

对于 `y = W @ x`（`y[o] = sum_j W[o,j] * x[j]`），当 W 和 x 都使用 FP4 量化时，高效的硬件实现需要一个关键条件：

> **W 和 x 在收缩维度（`j` 方向）上的 block 边界必须对齐，且在一个 block 内各自具有常数 scale。**

这样 FP4 的 multiply-accumulate 可以在 block 内全部完成（在更高精度如 INT32 中累加），scale 乘法推迟到 block 结束时才做一次。

### 7.2 数学分解

设有：
- W: `[out_features, in_features]`，沿 `in_features` 分 block（block_size = B），共 K 个 block
- x: `[tokens, in_features]`，同样沿 `in_features` 分 block

每个 block 内：
```text
y[o, t] = Σ_{k=0}^{K-1} ( s_w[o, k] · s_x[t, k] · Σ_{j in block_k} W_fp4[o,j] · x_fp4[t,j] )
```

- **内层求和** (FP4 × FP4 → INT32/FP16)：每 block 做 B 次 FP4 乘法和 (B-1) 次加法
- **外层乘法** (scale 乘法)：每 block 结束才做一次 double-scale 乘法
- **效率来源**：B 次低精度计算 → 1 次高精度 scale 乘法

### 7.3 量化布局要求

| 组件 | 量化粒度 | 分块方向 |
|------|---------|---------|
| **权重 W** | per-output-channel + per-in-feature-block | 沿 `in_features`（contraction dimension） |
| **激活 x** | per-token + per-in-feature-block | 沿 `in_features`（contraction dimension） |

**为什么 activation 必须是 per-token？**
不同 token 的动态范围差异巨大（如第一个 token 和最后一个 token 可能有数量级差异），共享 scale 会严重损害精度。

**为什么 weight 必须是 per-in-feature-block？**
如果把 weight flatten 到 1D 再分 block（旧实现），block 可能跨越多行，导致 block 边界不与 contraction 维度对齐，无法高效做 FP4 matmul。

### 7.4 代码对应关系

| 类 | 分块方式 | G 维度 | N（每 G 的 block 数） |
|---|---------|--------|---------------------|
| `NVFP4Quantization` (weight) | `[out_features, in_features]` → reshape → `[out_features, N, B]` | `out_features` | `in_features / B` |
| `NVFP4ActivationQuantization` (activation) | `[bs, n_seq, dim]` → reshape → `[bs*n_seq, dim]` → `[tokens, N, B]` | `bs × n_seq` | `dim / B` |

两者的 block 都沿 contraction 维度（最后一维），block_size 一致，因此 block 边界天然对齐。

### 7.5 与旧实现的区别

| 特性 | 旧 `NVFP4Quantization` | 新 `NVFP4Quantization` |
|------|----------------------|----------------------|
| 分块方式 | flatten 到 1D 后分块 | 沿 `in_features` 分块 |
| G 数量 | 1（全局一个 s_global） | `out_features`（每输出通道一个 s_global） |
| FP4 matmul 兼容 | **不兼容**（block 跨越行边界） | **兼容**（block 对齐 contraction 维度） |
| 精度 | 更粗（单一 s_global） | 更细（per-channel s_global） |

---

## 8. 实测：Sana Sprint 0.6B 的 FP4 量化误差分析

### 8.1 实验设置

| 项 | 值 |
|---|-----|
| 模型 | Sana Sprint 0.6B (1024px) |
| 层数 | 28 transformer blocks |
| head_dim | 32 |
| inner_dim | 1152 |
| token_dim (text) | 2304 |
| 量化参数 | `block_size=1536`, 仅权重量化（激活不量化） |
| 量化层数 | 464 个权重矩阵 |
| 测试输入 | latent=(1,32,32,32), text=(1,77,2304), t=500 |

### 8.2 总体误差

| 对比 | Cosine Sim | MSE | L1 | 相对误差 | Max Abs Err |
|------|:----------:|:---:|:---:|:--------:|:-----------:|
| Step1: diffusers(原始) vs our(未量化) | **0.99981** | 0.00050 | 0.0125 | 2.24% | 0.191 |
| Step2: our(未量化) vs our(NVFP4) | **0.95784** | 0.0940 | 0.256 | **45.4%** | 0.799 |
| Step3: diffusers(原始) vs our(NVFP4) | **0.95832** | 0.0963 | 0.258 | 46.3% | 0.909 |

**解读**：

- **Step 1 的 0.99981（非精确 1.0）**：2.24% 的相对误差可能来源于权重加载时的 dtype 转换差异（原始 bf16 重载为 fp32 后微小偏差）、或自定义实现与 diffusers 实现间存在非量化相关的结构差异（如 norm epsilon、attention 计算顺序等）。差异很小，不影响后续结论。
- **Step 2 的 Cosine 仅 0.9578**：仅权重量化（激活末量化）就导致输出 cos 下降 4.2%，说明 block_size=1536 的粗粒度量化对 0.6B 模型来说损伤严重。
- **Step 3 ≈ Step 2**：误差几乎完全由 FP4 量化主导，架构差异可忽略。

### 8.3 为什么 block_size=1536 效果很差

```
inner_dim = 1152,  block_size = 1536
→ num_blocks per channel = ceil(1152 / 1536) = 1
→ 每个 output channel 仅 1 个 s_block，需填充 384 个 zeros
```

 | 参数 | 值 |
 |------|-----|
 | in_features | 1152 |
 | block_size | 1536 |
 | padding | 384 (33.3% 是 zeros) |
 | N (每通道 block 数) | **1** |
 | s_global 数量 | out_features (per-channel) |
 | s_block 数量 | out_features × 1 (per-channel × 1) |

当 `N = 1` 时，两级缩放退化为**单级 per-channel 缩放 + FP4 量化**。整个通道的 1152 个元素共享一个 s_block，FP4 的 8 个正数值需要覆盖该通道的整个动态范围，精度损失剧烈。

### 8.4 Per-layer 误差分布（Top 10 / 464 层）

```
block.24.attn2.to_q.weight    MAE = 0.0504   ← cross-attention
block.24.attn2.to_k.weight    MAE = 0.0367
block.22.attn2.to_q.weight    MAE = 0.0352
block.23.attn2.to_q.weight    MAE = 0.0341
block.21.attn2.to_q.weight    MAE = 0.0328
block.16.attn2.to_k.weight    MAE = 0.0304
block.14.attn2.to_k.weight    MAE = 0.0300
block.12.attn2.to_k.weight    MAE = 0.0295
block.19.attn2.to_k.weight    MAE = 0.0292
block.20.attn2.to_q.weight    MAE = 0.0292
```

**规律**：

1. **Top 10 全部为 `attn2`（cross-attention）层**——这些权重将 text embeddings (2304d) 映射到 image latent space (1152d)，权重矩阵容易出现尖峰 outlier，在粗 block 量化下损伤最大。
2. **`to_q` 和 `to_k` 交替出现**，说明 cross-attention 的 query/key 投影质量化敏感度相近。
3. **深层 block（21-24）比浅层更敏感**，符合深层特征更精细、对量化更敏感的直觉。

### 8.5 不同 block_size 的理论精度预期

对于 inner_dim=1152 的模型：

| block_size | N per channel | s_block 粒度 | 预期 Cosine Sim | 推荐度 |
|:----------:|:------------:|:-----------:|:--------------:|:------:|
| 1536 | 1 | 单块覆盖全通道 | ~0.958 (实测) | ❌ 不可用 |
| 1152 | 1 | 恰好覆盖全通道 | ~0.958 | ❌ 无改善（padding 消除但仍是单块） |
| 576 | 2 | 通道一分为二 | ~0.97 | ⚠️ 改善有限 |
| 256 | 5 | 5 blocks / channel | ~0.98 | ✅ 可用 |
| 128 | 9 | 9 blocks / channel | ~0.985 | ✅ 推荐 |
| 64 | 18 | 18 blocks / channel | ~0.99 | ✅ 高精度 |
| 16 | 72 | 72 blocks / channel | ~0.995 | ✅ 最高精度（压缩比 ~7.1×） |

> **结论**：对于 inner_dim=1152 的模型，`block_size ≥ inner_dim` 等价于无 sub-channel 分块，不可接受。推荐 `block_size ≤ 256`（N ≥ 5），在压缩比和精度之间取平衡。

### 8.6 架构差异性排查（Step 1 为何不是 1.0）

可能来源：
- 原始 diffusers 权重是 `bfloat16`，加载后被 FFI 转为 `float32` —— 微量精度损失
- 自定义实现与 diffusers 在 attention 计算顺序、norm epsilon、残差连接位置等处可能有微小差异
- 2.24% 相对误差对该规模模型在可接受范围内，不足以影响量化误差的比较结论

---

## 9. 生成图片质量评估指标

当评估 FP4 量化对实际生成图片的影响时，不能仅看模型内部 layer 输出的数值误差，还需要在最终图像层面做感知质量对比。以下是四种常用指标的定义、范围和评价标准。

### 9.1 SSIM（结构相似性指数，Structural Similarity Index）

**定义**：从**亮度、对比度、结构**三个维度衡量两张图像素级相似度。

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

其中 $\mu$ 为均值（亮度），$\sigma$ 为标准差（对比度），$\sigma_{xy}$ 为协方差（结构）。

| 属性 | 说明 |
|------|------|
| **范围** | 0 ~ 1（1 = 完全相同） |
| **方向** | **↑ 越大越好** |
| **优秀** | ≥ 0.95 |
| **良好** | 0.85 ~ 0.95 |
| **一般** | 0.70 ~ 0.85 |
| **较差** | < 0.70 |

**适用范围**：对亮度/对比度变化敏感，但**对平移、旋转、轻微纹理变化不鲁棒**。适合检测"画面是否保持相同结构"，但两张语义相同构图略有偏移的图可能得分很低。

### 9.2 PSNR（峰值信噪比，Peak Signal-to-Noise Ratio）

**定义**：基于逐像素均方误差（MSE）的对数尺度信噪比。

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) = 20 \cdot \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right)$$

| 属性 | 说明 |
|------|------|
| **范围** | 0 ~ ∞ dB（通常 20~50 dB） |
| **方向** | **↑ 越大越好** |
| **优秀** | ≥ 40 dB（几乎无法察觉差异） |
| **良好** | 30 ~ 40 dB（差异极小） |
| **一般** | 20 ~ 30 dB（可察觉但不严重） |
| **较差** | < 20 dB（差异明显） |

**局限性**：纯数学指标，**不模拟人眼感知**。两张语义完全相同但平移 1 像素的图，PSNR 可能低至 20 dB，而人眼几乎看不出区别。在扩散模型评估中，PSNR 通常偏低是正常现象。

### 9.3 LPIPS（学习感知图像块相似度，Learned Perceptual Image Patch Similarity）

**定义**：用预训练深度网络（AlexNet/VGG/SqueezeNet）提取多层特征，在特征空间计算加权 L2 距离。被认为是**目前最接近人类感知判断**的图像相似度指标。

$$\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{y}_{hw}^l - \hat{y}_0^{l}_{hw}) \|_2^2$$

其中 $l$ 表示网络第 $l$ 层，$\hat{y}^l$ 为归一化后的特征激活，$w_l$ 为可学习权重。

| 属性 | 说明 |
|------|------|
| **范围** | 0 ~ ~0.7（0 = 完全相同） |
| **方向** | **↓ 越小越好** |
| **优秀** | < 0.10（几乎无法察觉） |
| **良好** | 0.10 ~ 0.20（轻微感知差异） |
| **一般** | 0.20 ~ 0.40（可见差异但不剧烈） |
| **较差** | > 0.40（明显不同） |

**优势**：LPIPS 对纹理、风格、语义变化敏感，能区分"像素不同但感知相同"与"像素不同且感知不同"的情况，比 SSIM/PSNR 更贴合人眼评价。

### 9.4 FID（Fréchet Inception Distance）

**定义**：用 InceptionV3 网络将图片映射到 2048 维特征空间，假设特征服从多元高斯分布，计算两个分布之间的 Fréchet 距离。

$$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

其中 $(\mu_r, \Sigma_r)$ 为参考图像集（真实图片）的特征统计量，$(\mu_g, \Sigma_g)$ 为生成图片集的特征统计量。

| 属性 | 说明 |
|------|------|
| **范围** | 0 ~ ∞（0 = 生成分布 = 真实分布） |
| **方向** | **↓ 越小越好** |
| **优秀** | < 10（高保真生成） |
| **良好** | 10 ~ 30 |
| **一般** | 30 ~ 60 |
| **较差** | > 60 |

**FID 的两种使用方式**：

| 方式 | 对比对象 | 含义 | 是否需要参考数据集 |
|------|---------|------|:-----------------:|
| **FID(生成, 真实)** | 生成图 vs COCO/ImageNet 等 | 生成质量离真实世界多远 | ✅ 需要 |
| **FID(FP4, Unquantized)** | FP4 生成图 vs 非量化生成图 | FP4 量化引入的分布偏移 | ❌ 不需要 |

**为什么 FID 需要 InceptionV3？**：直接比像素的话，两张语义相同但亮度不同的图会得到巨大的像素级差异。InceptionV3 的 pool3 层（2048 维）能将语义相近的图映射到相近的特征向量，在特征空间计算分布距离才有意义。

### 9.5 四项指标的对比总结

| 指标 | 衡量什么 | 优势 | 劣势 | 方向 |
|------|---------|------|------|:--:|
| **SSIM** | 结构/亮度/对比度相似 | 计算快，有理论基础 | 对平移/旋转敏感 | ↑ |
| **PSNR** | 像素级误差 | 最简单，广泛使用 | 不模拟人眼感知 | ↑ |
| **LPIPS** | 特征空间感知距离 | **最接近人眼判断** | 需加载预训练网络 | ↓ |
| **FID** | 图像分布距离 | 评估多样性和质量 | 需大量样本(≥10K 理想)，对预处理器敏感 | ↓ |

### 9.6 实测：FP4 量化的生成图像指标

以下为 Sana Sprint 0.6B（512px, 2 steps, 相同 seed 成对生成）的 FP4 vs 非量化对比：

| block_size | 图像对数 | SSIM ↑ | PSNR ↑ (dB) | LPIPS ↓ |
|:----------:|:--------:|:------:|:-----------:|:-------:|
| 256 | 5 | 0.7013 ± 0.080 | 19.48 ± 1.65 | 0.2709 ± 0.095 |
| 1536 | 4 | 0.7056 ± 0.061 | 19.16 ± 2.25 | 0.2803 ± 0.079 |

**解读**：
- 两个 block_size 下 SSIM/PSNR/LPIPS 几乎一致，差异在标准差范围内，**block_size 大小对最终生成图没有可测量的质量影响**
- SSIM ~0.70、LPIPS ~0.27 的指标水平，主要来自 2-step 低步数推理的随机性而非量化误差——扩散模型在极少步数下的输出天然存在波动
- 要提高指标区分度，建议增加 `steps` 和 `num_images`，或使用 FID(FP4, Unquantized) 直接在分布层面量化偏移

> **注意**：此处的生成质量指标与第 8 节的 layer-wise 数值误差衡量不同的东西。Layer-wise 误差（Cosine 0.958 at bs=1536）反映的是模型内部激活的偏离，而 SSIM/LPIPS 反映的是**经过多步扩散去噪后**最终图像的感知差异。扩散模型的多步迭代过程具有误差自校正能力，因此内部误差不一定会线性传导到最终图像。

### 9.7 评估脚本使用

```bash
# 快速评估（仅 SSIM/PSNR/LPIPS，skip 生成）
python scripts/eval_nvfp4_fid.py --skip_generation \
    --fp4_subdir fp4_bs1536 --output_suffix bs1536 --block_size 1536

# 加上 FID（FP4 vs 非量化，不需要参考数据集）
python scripts/eval_nvfp4_fid.py --skip_generation \
    --fp4_subdir fp4_bs1536 --output_suffix bs1536 --with_fid

# 完整生成 + 评估
python scripts/eval_nvfp4_fid.py --num_images 20 --steps 4 --resolution 1024 \
    --block_size 256 --with_fid --with_clip
```

# Evaluation Results Summary

**数据来源**: `G:\Outputs\Efficient-Diffusion\eval_gen`
**分析日期**: 2026-07-14
**模型**: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`
**数据集**: MJHQ-30K (30 images, sequential sampling, seed_start=42)
**公共配置**: steps=2, guidance_scale=4.5, resolution=1024

---

## 1. 实验结果总览

| 实验配置 | NVFP4 | Block Size | Rotation | Permutation | 图片数 | FID ↓ |
|---------|-------|------------|----------|-------------|--------|-------|
| **Sana_origin_sequential_30** | No | 256 | N/A | N/A | 30 | **269.41** |
| **Sana_nvfp4_bs16** | Yes | 16 | none | none | 30 | **274.05** |
| **Sana_nvfp4_bs16_had_mag** | Yes | 16 | hadamard | mag | 30 | **469.24** |
| Sana_origin | No | 256 | N/A | N/A | 12 | N/A (未评估) |

---

## 2. 各实验详细配置

### 2.1 Sana_origin_sequential_30 (基准)

| 参数 | 值 |
|------|-----|
| use_nvfp4 | `false` |
| block_size | 256 |
| seeds | 42-71 (每个图片递增) |
| batch_size | (默认) |
| FID | **269.41** |

原始 Sana 模型，不使用 NVFP4 量化。所有 30 张图片使用不同的 seed (42~71)，提供最公平的基准。

### 2.2 Sana_nvfp4_bs16

| 参数 | 值 |
|------|-----|
| use_nvfp4 | `true` |
| block_size | 16 |
| rotation | `none` |
| permutation | `none` |
| seed | 42 (所有图片) |
| batch_size | 2 |
| FID | **274.05** |

纯 NVFP4 量化（block_size=16），无旋转/置换。FID 仅比原始模型高 **4.64** 个点（+1.7%），说明 NVFP4 量化对图像质量的影响非常小。

### 2.3 Sana_nvfp4_bs16_had_mag (Hadamard + Magnitude Permutation)

| 参数 | 值 |
|------|-----|
| use_nvfp4 | `true` |
| block_size | 16 |
| rotation | `hadamard` |
| permutation | `mag` |
| seed | 42 (所有图片) |
| batch_size | 2 |
| FID | **469.24** |

Hadamard 旋转 + magnitude 置换。FID 为原始模型的 **1.74 倍**（恶化 199.83 个点），严重退化。


---

## 3. 对比分析与关键发现

### 3.1 NVFP4 量化的质量损失

```
Sana_origin → Sana_nvfp4_bs16:  FID 269.41 → 274.05  (+4.64, +1.7%)
```

NVFP4 block_size=16 的量化带来的 FID 损失仅为 **1.7%**，在可接受范围内。这表明该量化方案具有良好的保真度。

### 3.2 Hadamard + Mag 的严重退化

```
Sana_nvfp4_bs16 → Sana_nvfp4_bs16_had_mag:  FID 274.05 → 469.24  (+195.19, +71.2%)
Sana_origin → Sana_nvfp4_bs16_had_mag:       FID 269.41 → 469.24  (+199.83, +74.2%)
```

Hadamard 旋转 + magnitude 置换导致 FID 恶化约 **74%**，几乎翻倍。这与之前在 `analysis/` 中的分析结论一致：

- Hadamard 旋转虽然改善了 per-element 量化误差（分布更均匀 → NVFP4 block-wise scaling 更精确）
- 但旋转后的 `W_eff = R·W` 改变了量化误差的**空间结构**
- 误差从随机噪声变为与信号方向对齐的**相干误差**
- 经过矩阵乘法后，相干误差被系统性放大（误差放大效应）

### 3.3 与误差放大分析的一致性

之前的 analysis 结果显示：hadamard 配置下 **82-83%** 的 NVFP4Linear 层量化误差改善，但 output diff 改善率仅 **5-8%**。这表明：
- Per-element 误差变小 ✓
- Error propagation 反而变大 ✗

FID 退化 469.24 直接验证了上述分析的端到端影响：per-layer 的误差结构变化最终累积为显著的图像质量退化。

---

## 4. 建议

1. **NVFP4 纯量化 (`none_identity`)** 是最安全的配置，FID 损失仅 1.7%
2. **Hadamard + Mag** 配置需要谨慎：虽然 quant error 变小了，但 error propagation 变大了，最终 FID 严重退化
3. 后续可测试 `hadamard_identity` 或 `none_mag` 等其他组合，寻找量化误差与误差传播之间的最佳平衡点
4. 建议增加 samples 数量（当前仅 30），以获得更稳定的 FID 估算

---

## 5. 相关脚本

| 文件 | 功能 |
|------|------|
| `eval/generate_for_eval.py` | 在 MJHQ-30K 数据集上生成评测图片 |
| `eval/eval_comprehensive.py` | 综合评测：FID、CLIP Score、GenEval、DPG-Bench、ImageReward |
| `eval/eval_nvfp4_fid.py` | NVFP4 专项评测：量化 vs 非量化的 SSIM/PSNR/LPIPS/FID |

## 6. 使用示例

```powershell
# 1. 生成评测图片
python eval/generate_for_eval.py `
    --output_dir outputs/eval_gen/my_model `
    --num_images 500 --steps 4 --resolution 1024

# 2. 运行 FID 评测（预计算参考统计量后）
python eval/eval_comprehensive.py `
    --image_dir outputs/eval_gen/my_model --fid `
    --fid_ref_stats "G:/datasets/MJHQ-30K_fid_stats.npz"

# 3. 运行全部指标
python eval/eval_comprehensive.py `
    --image_dir outputs/eval_gen/my_model --all `
    --mjhq_path G:/datasets/MJHQ-30K
```

# Quantization Error vs Output Diff Analysis

**数据来源**: `G:\Outputs\Efficient-Diffusion\rot_perm_compare_module_steps_quantized`
**分析日期**: 2026-07-14
**Samples**: 1 张图, 2 decode timesteps, 9 种 rotation × permutation 配置

---

## 1. 核心问题

> 对大部分 NVFP4Linear 层，应用 rotation/permutation 后 **per-element 量化误差变小了，但 output diff（vs reference）变大了**，这是真的吗？

**答：是的，且模式非常显著。**

---

## 2. 用 MSE 指标统计（1856 个 layer × config 组合）

Baseline = `none_identity`（无旋转、恒等置换）

| 类别 | 数量 | 占比 | 含义 |
|------|------|------|------|
| **量化误差↓, output diff↑** | **680** | **36.6%** | **异常：量化改善了但输出反而变差** |
| 量化误差↓, output diff↓ | 223 | 12.0% | 良性：两者同向改善 |
| 量化误差↑, output diff↑ | 668 | 36.0% | 预期：两者同向变差 |
| 量化误差↑, output diff↓ | 285 | 15.4% | 反向：输出反而改善 |

**异常 case（680）是良性 case（223）的 3 倍**，确认了误差放大效应的存在。

---

## 3. 各配置逐项分析

| Config | quant 改善率 | output 改善率 | 异常层数 | 模式特点 |
|--------|-------------|-------------|---------|---------|
| **hadamard_identity** | **82%** | **6%** | **175** | 量化大幅改善，输出几乎全败 |
| **hadamard_random** | **83%** | **8%** | **175** | 同上 |
| **hadamard_mag** | 59% | 5% | 125 | 同上，程度稍轻 |
| none_mag | 36% | 54% | 32 | 输出改善为主 |
| none_random | 44% | 30% | 67 | 混合 |
| random_identity | 30% | 52% | 25 | 输出改善为主 |
| random_mag | 28% | 3% | 60 | 两者都差 |
| random_random | 27% | 61% | 21 | 输出改善为主 |

**Hadamard 旋转的三个配置是典型"罪魁祸首"**：82-83% 的层量化误差显著降低，但 output diff 改善率仅 5-8%，意味着绝大部分层的输出反而变差了。

---

## 4. 最异常的层（hadamard_identity vs none_identity）

| 层 | δ_param_mse | δ_act_mse | δ_output_mse | 输出恶化倍数 |
|----|------------|-----------|-------------|------------|
| block.1.attn2.to_out.0 | -4.7e-5 | +8e-6 | +0.063 | **17.9x** |
| time_embed.timestep_embedder.linear_2 | 0 | -1e-6 | +0.0003 | **15.6x** |
| time_embed.linear | -1.4e-5 | -1e-6 | +0.003 | **14.2x** |
| block.5.attn2.to_out.0 | -7.8e-5 | +1.3e-5 | +0.124 | **10.9x** |
| block.1.attn1.to_out.0 | -5e-5 | -3e-5 | +4.39 | **8.5x** |

这些层的 param/act 量化误差仅变化了 ~10⁻⁵ 量级，但 output diff 飙升 8-18 倍。

---

## 5. Cosine 指标的结论

用 cosine similarity 度量时，虽然绝对数值模式不同，但异常 case（199）仍然 dominate 良性 case（64），比例约 3:1。

| 类别 | 数量 | 占比 |
|------|------|------|
| 量化误差↓, output diff↓ (反常) | **199** | **10.7%** |
| 量化误差↓, output diff↑ (良性) | 64 | 3.4% |
| 量化误差↑, output diff↑ (预期) | 1180 | 63.6% |
| 量化误差↑, output diff↓ (反向) | 413 | 22.3% |

---

## 6. 物理原因

对于线性层 \( y = W \cdot x \)：

\[
\hat{y} = (W + \delta W) \cdot (x + \delta x)
\]

\[
\hat{y} - y = W \cdot \delta x + \delta W \cdot x + \delta W \cdot \delta x
\]

per-element MSE 测量的是：

- `param_mse ≈ E[δW²]` — 直接权重量化误差
- `act_mse ≈ E[δx²]` — 直接激活量化误差

但 output error 包含：

- `E[(W·δx)²] ≈ (||W||²_F / d_in) · act_mse`
- `E[(δW·x)²] ≈ (||x||² / d_in) · param_mse`

放大因子 `||W||²_F/d_in` 和 `||x||²/d_in` 通常 >> 1（可达 100x–10000x）。

**Hadamard 旋转之所以"反常"**：
- 旋转使 W 的数值分布更均匀 → NVFP4 block-wise 量化更精确 → per-element δW, δx 变小
- 但旋转后的 `W_eff = R·W` 改变了误差的**结构**：δW 不再是随机噪声，而是与信号方向对齐的相干误差
- 经过矩阵乘法 `δW·x` 后，相干误差被系统性放大，远超 per-element 层面的改善

简言之：**per-element error 变小 ≠ error propagation 变小**。

---

## 7. 相关脚本

| 文件 | 功能 |
|------|------|
| `analysis/cal_rotation_permutation_quantization_error.py` | 主计算脚本：在多配置下运行 decode loop，收集 param/act/output 误差 |
| `analysis/visualization_rotation_permutation_quantization_error.py` | 可视化脚本：绘制 layer_error_grid 和 final_diff_grid |
| `analysis/analyze_param_act_vs_output.py` | 本文档的分析脚本：统计异常层比例、输出放大因子 |

---

## 8. 使用示例

```powershell
# 运行计算（收集量化误差数据）
python analysis/cal_rotation_permutation_quantization_error.py `
    --dataset_path G:\datasets\MJHQ-30K --n_samples 1 `
    --output_dir G:\Outputs\Efficient-Diffusion\rot_perm_compare `
    --rots none,hadamard,random --perms identity,random,mag `
    --quantized_modes quantized --cosine --decode_steps 2 --generate_images

# 可视化（plot 热力图）
python analysis/visualization_rotation_permutation_quantization_error.py `
    --input-dir G:\Outputs\Efficient-Diffusion\rot_perm_compare `
    --metric mse --baseline none_identity --logy

# 统计分析（确认异常层比例）
python analysis/analyze_param_act_vs_output.py `
    --input-dir G:\Outputs\Efficient-Diffusion\rot_perm_compare `
    --metric mse --baseline none_identity
```

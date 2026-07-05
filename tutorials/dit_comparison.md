# HunyuanDiT vs SD3: Architecture & Performance Comparison

## 1. Architecture Differences

Both are Diffusion Transformers (DiT), but their core design philosophies diverge significantly.

| Dimension              | **SD3**                                                  | **HunyuanDiT**                                                    |
|------------------------|----------------------------------------------------------|-------------------------------------------------------------------|
| **Transformer Arch**   | **MMDiT** — separate weights for image & text streams, joint attention | Traditional DiT — single Transformer, image attends to text via Cross-Attention |
| **Attention**          | **Joint Attention**: img Q attends to concat([img_KV, txt_KV]); text also has independent QKV | **Cross-Attention**: img Q attends to text KV (encoder-decoder pattern) |
| **Position Encoding**  | 2D **Sinusoidal** (fixed, non-learnable)                 | **RoPE** (Rotary Position Embedding), applied only to image Q/K   |
| **Condition Modulation** | **adaLN-Zero**: 6-dim scale/shift/gate (3 for img stream + 3 for txt stream) | **adaLN + Style Cond**: injects Image Pooled Embedding as style condition; no shared img/txt modulation |
| **Text Encoder**       | CLIP-L + CLIP-G (concatenated on channel dim)            | **Bilingual CLIP** (1024d) + **Multilingual T5** (2048d), concatenated along sequence dim |
| **Block Design**       | Dual-stream FFN (img FFN + txt FFN), each with independent norm + FF | Single flow: Self-Attn → Cross-Attn → FFN; **long skip connections** between blocks |
| **Parameters**         | SD3 Medium: 2B, SD3 Large: 8B                           | ~1.5B (v1.2)                                                      |
| **Training Framework** | Flow Matching                                            | DDPM / Flow Matching (optional)                                   |
| **Multi-Resolution**   | Limited resolution generalization                       | Native Multi-Resolution support (position encoding interpolation) |

### Block Diagram

```
  SD3 MMDiT Block                    HunyuanDiT Block
  ┌─────────────────────┐           ┌─────────────────────┐
  │  img             txt │           │       Input          │
  │   │               │  │           │         │            │
  │  adaLN-Zero  adaLN-Zero         │    Self-Attn (RoPE)  │
  │   │               │  │           │         │            │
  │   └── Joint Attn ──┘  │           │  Cross-Attn (txt)   │
  │   │               │  │           │         │            │
  │  gate_msa    gate_msa│           │  adaLN + Style Cond  │
  │   │               │  │           │         │            │
  │  img FFN      txt FFN│           │        FFN           │
  │   │               │  │           │         │            │
  │  gate_mlp    gate_mlp│          │    + Skip Connect     │
  └─────────────────────┘           └─────────────────────┘
```

**Key difference**: SD3 gives image and text embeddings independent weight spaces, interacting through Joint Attention. HunyuanDiT treats text embedding only as K/V in Cross-Attention — the model body processes image tokens only.

### Component Size Breakdown

| Component            | HunyuanDiT         | SD3 Medium          |
|----------------------|--------------------|---------------------|
| Transformer (DiT)    | ~5.6 GB (28 layers)| ~4.7 GB (24 layers) |
| text_encoder         | ~1.3 GB (Bilingual CLIP) | ~1.3 GB (CLIP-L+G) |
| text_encoder_2       | ~6.2 GB (mT5, optional) | ~9 GB (T5-XXL, optional) |
| VAE                  | ~319 MB (SDXL VAE) | ~319 MB (SDXL VAE)  |
| **Total (full)**     | **~13.4 GB**        | **~15.3 GB**        |
| **Total (no T5)**    | **~7.2 GB**         | **~6.3 GB**         |

---

## 2. Performance Comparison

Human evaluation from the HunyuanDiT paper (4 dimensions, 50+ professional evaluators):

| Model                    | Text-Image Consistency | Excluding AI Artifacts | Subject Clarity | Aesthetics | **Overall Pass Rate** |
|--------------------------|------------------------|------------------------|-----------------|------------|-----------------------|
| DALL-E 3 (closed-source) | 83.9                   | 80.3                   | 96.5            | 89.4       | **71.0%**             |
| MidJourney v6 (closed)   | 73.5                   | 80.2                   | 93.5            | 87.2       | **63.3%**             |
| **HunyuanDiT** (open)    | **74.2**               | **74.3**               | **95.4**        | **86.6**   | **59.0%**             |
| SD3 (closed at time)     | 77.1                   | 69.3                   | 94.6            | 82.5       | **56.7%**             |
| Playground 2.5           | 71.9                   | 70.8                   | 94.9            | 83.3       | 54.3%                 |
| PixArt-α                 | 68.3                   | 60.9                   | 93.2            | 77.5       | 45.5%                 |
| SDXL                     | 64.3                   | 60.6                   | 91.1            | 76.3       | 42.7%                 |

**Key takeaways:**

- **HunyuanDiT ranks #1 among open-source models** (59.0% vs SD3 56.7%), especially strong in Subject Clarity (95.4%) and Aesthetics (86.6%).
- SD3 leads slightly in **Text-Image Consistency** (77.1% vs 74.2%) — the MMDiT joint attention advantage.
- HunyuanDiT has **absolute advantage in Chinese understanding** (Bilingual CLIP + Multilingual T5).
- SD3's MMDiT dual-stream design has more parameters (2B vs 1.5B), offering stronger theoretical modeling capacity at the cost of higher training resources.

### Ablation Insights (from HunyuanDiT paper)

- **Skip connections**: Removing long skip connections hurts both FID and CLIP score.
- **Position encoding**: RoPE outperforms sinusoidal encoding and accelerates convergence; adding RoPE to text features yields no gain.
- **Text encoder**: Bilingual CLIP + Multilingual T5 significantly outperforms CLIP or T5 alone; sequence-dim concatenation beats channel-dim concatenation.

---

## 3. When to Use Which

| Scenario                            | Recommendation      | Reason                                    |
|-------------------------------------|---------------------|-------------------------------------------|
| Chinese text-to-image               | **HunyuanDiT**      | Native bilingual support, Chinese cuisine/poetry/culture |
| English high-quality generation     | Either              | SD3 slightly better text-image alignment  |
| Resource-constrained / edge deploy  | **HunyuanDiT**      | 1.5B params; can skip T5, saving ~6 GB    |
| Text rendering / spelling tasks     | SD3                 | MMDiT models text structure better        |
| Architecture research / hacking     | SD3                 | MMDiT is a more cutting-edge direction    |
| Multi-resolution generation         | **HunyuanDiT**      | Native multi-resolution with RoPE interpolation |

---

## 4. References

- HunyuanDiT paper: [arXiv 2405.08748](https://arxiv.org/abs/2405.08748)
- HunyuanDiT GitHub: [Tencent-Hunyuan/HunyuanDiT](https://github.com/Tencent-Hunyuan/HunyuanDiT)
- SD3 technical report: [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3-research-paper)
- HunyuanDiT on ModelScope: `dengcao/HunyuanDiT-v1.2-Diffusers`
- SD3 on ModelScope: `AI-ModelScope/stable-diffusion-3-medium-diffusers`

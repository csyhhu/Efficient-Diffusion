# Step-Wise Difference Analysis

Analyze the **step-to-step differences** within a single denoising trajectory.
The goal is to understand how the DiT output and noise prediction evolve
across denoising steps — e.g. whether certain steps produce similar outputs
(suggesting redundancy) or whether early/late steps dominate the generation.

## 1. Save step-wise outputs

Use `save_step_wise_output.py` to generate images and save per-step
`dit_outputs` / `noise_preds` for each sample:

```bash
python -m analysis.step_wise_difference.save_step_wise_output `
    --model_id "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" `
    --dataset_name mjhq30k --dataset_path "G://datasets/MJHQ-30K" `
    --n_samples 10 --num_steps 2 --seed 42 `
    --output_dir "G://Outputs//Efficient-Diffusion//step_wise_output//Sana-MJHQ30K"
```

Each run produces:
- `metadata.json` — hyperparameters and model identity
- `00000.png`, `00001.png`, … — final generated images (saved by `generate`)
- `sample_0000.pt`, `sample_0001.pt`, … — per-step intermediates with keys:
  - `dit_outputs`: list of raw transformer outputs at each step
  - `noise_preds`: list of noise predictions after remap / CFG merge

## 2. Analyze step-wise differences

Use `analyze_step_wise_difference.py` to compute three modules of analysis
on the saved `.pt` files:

```bash
python -m analysis.step_wise_difference.analyze_step_wise_difference `
    --input_dir "G://Outputs//Efficient-Diffusion//step_wise_output//Sana-MJHQ30K-nvfp4" `
    --sample_idx 0
```

### Analysis modules

**Module A — Pairwise relative L2 distance matrix:**
For each pair of steps (i, j), compute `||o_i - o_j|| / ||o_i||`.
This reveals which steps produce similar outputs and which diverge.
A multi-sample mean heatmap and a std heatmap (cross-sample consistency)
are generated, plus an optional single-sample heatmap.

**Module B — Step-wise L2 norm curve:**
Plot the L2 norm of each step's output, averaged over all samples with
±1 std error band. This shows whether the output magnitude is stable
across steps or grows/shrinks.

**Module C — Accumulation error:**
For each step i, compute `||N * o_i - sum(o_j)|| / ||sum(o_j)||`.
This measures how well a single step's output represents the entire
trajectory. The best step (lowest error) indicates the most "average"
output. Includes both multi-sample mean ± std and single-sample overlay.

Each module is applied to both `dit_outputs` and `noise_preds`.

### Output

Analysis plots and a `summary.json` are saved to `--output_dir`
(defaults to `<input_dir>/analysis/`).

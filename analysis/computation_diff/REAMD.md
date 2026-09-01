This document analyzes the computation diff of the Diffusion model: how much a token is modified before and after a computation unit (it could be a step, layer, or module (self-attention, ffn and etc)).

# How to use
## Analysis
Computation diff generation, analysis, visualization and skip plan generation for a DiT model.
```bash
python -m analysis.computation_diff.save_analyze_computation_diff
```
## Generation with Token Skip
Given a skip plan:
```bash
python -m analysis.computation_diff.test_token_skip `
    --skip_plan_path G://Outputs//Efficient-Diffusion//computation_diff//SD3-MJHQ30K//skip_plan/full-steps-cos-step-aggr-global-50.json `
    --postfix "cos-step-aggr-global-50"
```

# Some Conclusions
- Computation diff may not be a good criterion for token skip.
- Step-wise different in `../step_wise_difference` is more useful in identifying important layers.
Light weight toolkit and experiment playground for efficient diffusion

# Quick Start

```bash
pip install -r requirements.txt
```

# Walkthrough of DDPM and FM
```python
# Training and sample a DDPM using MNIST dataset
python -m scripts.mnist_train_ddpm
```

```python
# Training and sample a FM using MNIST dataset
python -m scripts.mnist_train_fm
```

# Unified Training & Sampling for Full-Precision/Quantized using DDPM/FM/CM on Various Benchmark
```python
# Training a Quantized Simple DiT on MNIST dataset using Flow Matching
python -m main `
    --model_name=quantized_dit `
    --model_config_path=config/mnist_dit_fm/model.yaml `
    --dataset_name=MNIST `
    --dataset_config_path=config/mnist_dit_fm/dataset.yaml `
    --running_config_path=config/mnist_dit_fm/running.yaml `
    --output_dir=Results/mnist_quantized_dit_fm
```

# NVFP4
## New Model Download
Download a new model (especially its parameters) and validate.
```bash
python -m test.test_model_download_generation
```

## Test ImageGenerator in Newly Download Model
Test whether ImageGenerator is compatible with newly downloaded model.
```bash
python -m test.test_image_generation
```

## Rotation / Permutation Test
After NVFP4 Quantized version of model is generated, test its correctness.
```bash
python -m test.test_rotation_permutation
```

## Conduct Caylay Rotation Calibration
Conudct caylay rotation calibration and save the generated model.
```bash
python -m scripts.caylay_calibration --save_root G:/Outputs/Efficient-Diffusion/ckpt/cayley_rotation
```

# Step-Wise Difference Analysis

Analyze step-to-step differences within a denoising trajectory. See
[analysis/step_wise_difference/READMD.md](analysis/step_wise_difference/READMD.md)
for details.

## Save step-wise outputs
```bash
python -m analysis.step_wise_difference.save_step_wise_output --model_type sana --model_id "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" --dataset_name mjhq30k --dataset_path "G://datasets/MJHQ-30K" --n_samples 10 --num_steps 4 --seed 42 --guidance 4.5 --use_nvfp4 --output_dir "G://Outputs//Efficient-Diffusion//step_wise_output//Sana-MJHQ30K-nvfp4"
```

## Analyze
```bash
python -m analysis.step_wise_difference.analyze_step_wise_difference --input_dir "G://Outputs//Efficient-Diffusion//step_wise_output//Sana-MJHQ30K-nvfp4" --sample_idx 0
```

# Evaluation

## Generate Images for Evaluation
For `identity`, `random`, `hadamard` rotation and `identity`, `random`, `hadamard` permutation:
```bash
python -m eval.generate_for_eval `
    --model_id stabilityai/stable-diffusion-3.5-medium `
    --quantized --rotation hadamard --permutation identity `
    --dataset_name coco2017val `
    --dataset_path G://datasets//coco2017val `
    --save_root G://Outputs//Efficient-Diffusion//eval_gen/sana-coco2017val-hadamard-identity
```
For `cayley` rotation:
```bash
python -m eval.generate_for_eval `
    --model_id stabilityai/stable-diffusion-3.5-medium `
    --rotation_ckpt_path G://Outputs//Efficient-Diffusion//ckpt/cayley_rotation `
    --quantized  --rotation cayley `
    --dataset_name coco2017val `
    --dataset_path G://datasets//coco2017val `
    --save_root G://Outputs//Efficient-Diffusion//eval_gen/sana-coco2017val-cayley
```

## FID
Generate real images' FID stats as baseline.
```bash
# coco2017 FID stats
python -m eval.main --fid --dataset_name coco2017 `
    --precompute_fid_stats `
    --fid_ref_stats G://datasets/coco2017_fid_stats.npz

# 计算生成图 FID
```

Calculate generated images' FID stats.
```bash
python -m eval.main --fid `
    --input_dir G://Outputs//Efficient-Diffusion//eval_gen/sana-coco2017val-hadamard-identity `
    --output_dir G://Outputs//Efficient-Diffusion//eval_gen/sana-coco2017val-hadamard-identity `  
    --dataset_name coco2017 `
    --fid_ref_stats G://datasets/coco2017_fid_stats.npz
```

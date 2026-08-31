# Attention Pattern Analysis

This document analyzes the attention pattern of the Diffusion model. 

It first generates the attention map for a serval images for each layer, each timestamp and each head. Then, it analyzes the attention pattern by visualizing the attention map.

It also analyzes the token variation during generation: whether there exists a large portion of tokens whose representation changes little. If so, its computation can be skipped. It first calcualtes the difference between before and afer attention computation, 

## Attention Map Generation

## Attention Map Visualization
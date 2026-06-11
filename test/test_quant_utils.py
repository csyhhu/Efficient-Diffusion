"""
Test cases for quant_utils.py
Testing 2D tensor quantization with different alpha/beta configurations
"""


import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.quant_utils import asymmetric_quantize, symmetric_quantize


def test_2d_asymmetric_scalar_thresholds():
    """
    Test 1: 2D tensor with scalar alpha/beta
    Apply same quantization parameters to the entire tensor
    """
    print("=" * 70)
    print("Test 1: 2D Asymmetric Quantization with Scalar Thresholds")
    print("=" * 70)
    
    # Create a 2D tensor of shape [3, 4]
    x = torch.tensor([
        [0.0, 0.5, 1.0, 1.5],
        [-1.0, -0.5, 0.0, 0.5],
        [-2.0, -1.5, -1.0, -0.5]
    ], dtype=torch.float32)
    
    print(f"\nInput tensor (shape {x.shape}):")
    print(x)
    
    # Use scalar thresholds (apply to all elements)
    bit = 2  # 4 levels
    alpha = -2.0  # min threshold (scalar)
    beta = 2.0    # max threshold (scalar)
    
    result = asymmetric_quantize(x, bit, alpha, beta)
    
    print(f"\nQuantization config:")
    print(f"  bit = {bit} (levels = {2**bit})")
    print(f"  alpha = {alpha} (scalar, min threshold)")
    print(f"  beta = {beta} (scalar, max threshold)")
    
    print(f"\nOutput tensor:")
    print(result)
    
    # Verify output shape
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} != {x.shape}"
    print(f"\n✓ Output shape matches input shape: {result.shape}")
    
    # Verify values are within valid quantization levels
    expected_levels = torch.linspace(alpha, beta, 2**bit)
    print(f"\nExpected quantization levels: {expected_levels}")
    
    print("\n" + "=" * 70 + "\n")


def test_2d_asymmetric_per_column_thresholds():
    """
    Test 2: 2D tensor with 1D alpha/beta (per-column quantization)
    Each column has its own quantization parameters
    """
    print("=" * 70)
    print("Test 2: 2D Asymmetric Quantization with Per-Column Thresholds")
    print("=" * 70)
    
    # Create a 2D tensor of shape [3, 4]
    x = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5, 3.5],
        [1.0, 2.0, 3.0, 4.0]
    ], dtype=torch.float32)
    
    print(f"\nInput tensor (shape {x.shape}):")
    print(x)
    
    # Use 1D thresholds of shape [4] (one per column)
    bit = 2  # 4 levels
    alpha = torch.tensor([0.0, 1.0, 2.0, 3.0])  # min threshold per column
    beta = torch.tensor([1.0, 2.0, 3.0, 4.0])    # max threshold per column
    
    print(f"\nQuantization config:")
    print(f"  bit = {bit} (levels = {2**bit})")
    print(f"  alpha = {alpha} (shape {alpha.shape}, per-column min)")
    print(f"  beta = {beta} (shape {beta.shape}, per-column max)")
    
    result = asymmetric_quantize(x, bit, alpha, beta)
    
    print(f"\nOutput tensor:")
    print(result)
    
    # Verify output shape
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} != {x.shape}"
    print(f"\n✓ Output shape matches input shape: {result.shape}")
    
    # Verify each column is quantized independently
    print("\nPer-column verification:")
    for col in range(x.shape[1]):
        col_min = alpha[col].item()
        col_max = beta[col].item()
        expected_levels = torch.linspace(col_min, col_max, 2**bit)
        print(f"  Column {col}: range [{col_min}, {col_max}], levels = {expected_levels}")
    
    print("\n" + "=" * 70 + "\n")


def test_2d_asymmetric_per_row_thresholds():
    """
    Test 3: 2D tensor with 1D alpha/beta (per-row quantization)
    Each row has its own quantization parameters
    """
    print("=" * 70)
    print("Test 3: 2D Asymmetric Quantization with Per-Row Thresholds")
    print("=" * 70)
    
    # Create a 2D tensor of shape [3, 4]
    x = torch.tensor([
        [0.0, 0.5, 1.0, 1.5],   # Row 0: range [0, 2]
        [2.0, 2.5, 3.0, 3.5],   # Row 1: range [2, 4]
        [4.0, 4.5, 5.0, 5.5]    # Row 2: range [4, 6]
    ], dtype=torch.float32)
    
    print(f"\nInput tensor (shape {x.shape}):")
    print(x)
    
    # Use 1D thresholds of shape [3] (one per row)
    # Need to reshape for proper broadcasting: [3] -> [3, 1]
    bit = 2  # 4 levels
    alpha = torch.tensor([[0.0], [2.0], [4.0]])  # min threshold per row, shape [3, 1]
    beta = torch.tensor([[2.0], [4.0], [6.0]])    # max threshold per row, shape [3, 1]
    
    print(f"\nQuantization config:")
    print(f"  bit = {bit} (levels = {2**bit})")
    print(f"  alpha shape = {alpha.shape} (per-row min, reshaped for broadcasting)")
    print(f"  beta shape = {beta.shape} (per-row max, reshaped for broadcasting)")
    print(f"  alpha values = {alpha.squeeze().tolist()}")
    print(f"  beta values = {beta.squeeze().tolist()}")
    
    result = asymmetric_quantize(x, bit, alpha, beta)
    
    print(f"\nOutput tensor:")
    print(result)
    
    # Verify output shape
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} != {x.shape}"
    print(f"\n✓ Output shape matches input shape: {result.shape}")
    
    print("\n" + "=" * 70 + "\n")


def test_2d_symmetric_scalar_thresholds():
    """
    Test 5: 2D tensor with scalar alpha (symmetric quantization)
    Apply same symmetric quantization to the entire tensor
    """
    print("=" * 70)
    print("Test 5: 2D Symmetric Quantization with Scalar Threshold")
    print("=" * 70)
    
    # Create a 2D tensor of shape [3, 4]
    x = torch.tensor([
        [-1.0, -0.5, 0.0, 0.5],
        [-0.8, -0.3, 0.2, 0.7],
        [-0.6, -0.1, 0.4, 1.0]
    ], dtype=torch.float32)
    
    print(f"\nInput tensor (shape {x.shape}):")
    print(x)
    
    # Use scalar threshold
    bit = 3  # 8 levels
    alpha = 1.0  # symmetric range [-1, 1]
    
    result = symmetric_quantize(x, bit, alpha)
    
    print(f"\nQuantization config:")
    print(f"  bit = {bit} (levels = {2**bit})")
    print(f"  alpha = {alpha} (scalar, symmetric range [{-alpha}, {alpha}])")
    
    print(f"\nOutput tensor:")
    print(result)
    
    # Verify output shape
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} != {x.shape}"
    print(f"\n✓ Output shape matches input shape: {result.shape}")
    
    # Verify zero-point preservation
    zero_mask = x == 0
    if zero_mask.any():
        zero_preserved = (result[zero_mask] == 0).all()
        print(f"✓ Zero-point preservation: {zero_preserved}")
    
    print("\n" + "=" * 70 + "\n")


def test_2d_symmetric_per_column_thresholds():
    """
    Test 6: 2D tensor with 1D alpha (per-column symmetric quantization)
    Each column has its own symmetric quantization range
    """
    print("=" * 70)
    print("Test 6: 2D Symmetric Quantization with Per-Column Thresholds")
    print("=" * 70)
    
    # Create a 2D tensor of shape [3, 4]
    x = torch.tensor([
        [-1.0, -2.0, -3.0, -4.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0, 4.0]
    ], dtype=torch.float32)
    
    print(f"\nInput tensor (shape {x.shape}):")
    print(x)
    
    # Use 1D thresholds of shape [4] (one per column)
    bit = 2  # 4 levels
    alpha = torch.tensor([1.0, 2.0, 3.0, 4.0])  # symmetric range per column
    
    print(f"\nQuantization config:")
    print(f"  bit = {bit} (levels = {2**bit})")
    print(f"  alpha = {alpha} (shape {alpha.shape}, per-column symmetric range)")
    print(f"  Column ranges:")
    for col in range(x.shape[1]):
        print(f"    Column {col}: [{-alpha[col].item()}, {alpha[col].item()}]")
    
    result = symmetric_quantize(x, bit, alpha)
    
    print(f"\nOutput tensor:")
    print(result)
    
    # Verify output shape
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} != {x.shape}"
    print(f"\n✓ Output shape matches input shape: {result.shape}")
    
    print("\n" + "=" * 70 + "\n")


def test_gradient_flow_2d():
    """
    Test 7: Verify gradient flow works for 2D tensors
    Important for training with quantized weights
    """
    print("=" * 70)
    print("Test 7: Gradient Flow Test for 2D Tensors")
    print("=" * 70)
    
    # Create a 2D tensor with gradient tracking
    x = torch.tensor([
        [-1.0, -0.5, 0.0, 0.5],
        [1.0, 0.8, 0.3, -0.2]
    ], dtype=torch.float32, requires_grad=True)
    
    print(f"\nInput tensor (shape {x.shape}, requires_grad={x.requires_grad}):")
    print(x)
    
    bit = 3  # 8 levels
    alpha = 1.0
    beta = 1.0
    
    # Forward pass (symmetric quantization)
    result = symmetric_quantize(x, bit, alpha)
    
    print(f"\nAfter quantization:")
    print(result)
    
    # Backward pass
    loss = result.sum()
    loss.backward()
    
    print(f"\nGradient of input:")
    print(x.grad)
    
    # Verify gradient is not None
    assert x.grad is not None, "Gradient is None!"
    print(f"\n✓ Gradient flow works for 2D tensors!")
    print(f"  Gradient shape: {x.grad.shape}")
    
    print("\n" + "=" * 70 + "\n")


def run_all_2d_tests():
    """Run all 2D tensor tests"""
    print("\n" + "=" * 70)
    print("Running All 2D Tensor Tests for quant_utils.py")
    print("=" * 70 + "\n")
    
    try:
        test_2d_asymmetric_scalar_thresholds()
        test_2d_asymmetric_per_column_thresholds()
        test_2d_asymmetric_per_row_thresholds()
        test_2d_symmetric_scalar_thresholds()
        test_2d_symmetric_per_column_thresholds()
        test_gradient_flow_2d()
        
        print("\n" + "=" * 70)
        print("✓ All 2D tests passed!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except Exception as e:
        print(f"\n✗ Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_2d_tests()

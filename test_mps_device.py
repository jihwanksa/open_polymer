#!/usr/bin/env python3
"""
Test script to verify MPS (Apple Silicon) GPU acceleration is available
Run with: conda activate polymer && python test_mps_device.py
"""

import torch

print("="*70)
print("APPLE SILICON (MPS) DEVICE DETECTION TEST")
print("="*70)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print()

# Auto-detect best device
if torch.cuda.is_available():
    device = 'cuda'
    device_name = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    device = 'mps'
    device_name = "Apple Silicon (MPS)"
else:
    device = 'cpu'
    device_name = "CPU"

print(f"✓ Selected Device: {device} ({device_name})")
print()

# Test tensor creation
try:
    test_tensor = torch.randn(5, 5).to(device)
    print(f"✓ Successfully created tensor on {device}")
    print(f"✓ Tensor device: {test_tensor.device}")
    print()
    
    # Test some operations
    result = torch.matmul(test_tensor, test_tensor.T)
    print(f"✓ Matrix multiplication works on {device}")
    print(f"✓ Result device: {result.device}")
    print()
    
    if device == 'mps':
        print("🎉 Great! Your M-series Mac will use Apple Silicon for GNN training!")
        print("   Expected speedup: 2-5x faster than CPU for neural networks")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("="*70)


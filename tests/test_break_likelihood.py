"""Test break_likelihood module."""

import sys
sys.path.insert(0, 'c:/Users/Andrew/projects/adia_structural_break')

print("Testing break_likelihood module...")
print("=" * 70)

# Run self-check
import src.sb.features.break_likelihood

print("\n" + "=" * 70)
print("✓ Module loaded successfully!")
print("\nUsage examples:")
print("\n1. Train baseline with break-likelihood:")
print("   python scripts/train_local.py --mode baseline --break-likelihood")
print("\n2. Train baseline with all features (multiscale+spectral+wavelet):")
print("   python scripts/train_local.py --mode baseline --multiscale --spectral --wavelet --break-likelihood")
print("\n3. Diagnostic with break-likelihood:")
print("   python scripts/diagnostic_baseline.py --multiscale --spectral --wavelet --break-likelihood")
print("\n4. Inference with break-likelihood:")
print("   python scripts/infer_local.py --mode baseline --multiscale --spectral --wavelet --break-likelihood")
print("=" * 70)

"""
Quick test of Phase 1 features.

Tests the new features on a small subset to verify they work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from sb import data_loader
from sb.features import cv_features, transformations, compression, cusum

print("="*70)
print("PHASE 1 FEATURES TEST")
print("="*70)

# Load small subset
print("\nLoading data...")
df, y = data_loader.load_for_training("data")
series_ids = df['id'].unique()[:3]  # Test on 3 series

for series_id in series_ids:
    series_data = df[df['id'] == series_id]
    x0 = series_data[series_data['period'] == 0]['value'].values
    x1 = series_data[series_data['period'] == 1]['value'].values
    
    print(f"\n--- Series {series_id} ---")
    print(f"Pre-break length: {len(x0)}, Post-break length: {len(x1)}")
    
    # Test CV features
    print("\n1. Testing CV features...")
    cv_feats = cv_features.compute_cv_features(x0, x1)
    print(f"   Generated {len(cv_feats)} CV features")
    print(f"   cv_global: {cv_feats['cv_global']:.4f}")
    print(f"   cv_global_easy_neg: {cv_feats['cv_global_easy_neg']}")
    print(f"   cv_std_interaction: {cv_feats['cv_std_interaction']:.4f}")
    
    # Test transformations
    print("\n2. Testing transformations...")
    for transform in ['raw', 'cumsum', 'diff']:
        x0_t, x1_t = transformations.transform_series(x0, x1, transform)
        print(f"   {transform:8s}: x0={len(x0_t):4d}, x1={len(x1_t):4d}")
    
    # Test compression
    print("\n3. Testing compression features...")
    comp_feats = compression.compute_compression_features(x0, x1)
    print(f"   Generated {len(comp_feats)} compression features")
    print(f"   zlib_pre: {comp_feats['zlib_pre']:.4f}")
    print(f"   zlib_post: {comp_feats['zlib_post']:.4f}")
    print(f"   lz_complexity_pre: {comp_feats['lz_complexity_pre']}")
    print(f"   ncd_pre_post: {comp_feats['ncd_pre_post']:.4f}")
    
    # Test CUSUM
    print("\n4. Testing CUSUM features...")
    cusum_feats = cusum.compute_cusum_features(x0, x1)
    print(f"   Generated {len(cusum_feats)} CUSUM features")
    print(f"   cusum_global_jump: {cusum_feats['cusum_global_jump']:.4f}")
    print(f"   elbow_category: {cusum_feats['elbow_category']}")
    print(f"   elbow_sharpness: {cusum_feats['elbow_sharpness']:.4f}")
    print(f"   cusum_error_wasserstein: {cusum_feats['cusum_error_wasserstein']:.4f}")

print("\n" + "="*70)
print("✅ ALL PHASE 1 FEATURES WORKING!")
print("="*70)
print("\nNext step: Run training with Phase 1 features:")
print("  python scripts/train_local.py --multiscale --phase1")
print("\nOr individually:")
print("  python scripts/train_local.py --multiscale --cv")
print("  python scripts/train_local.py --multiscale --cv --transforms")
print("  python scripts/train_local.py --multiscale --cv --transforms --compression --cusum")

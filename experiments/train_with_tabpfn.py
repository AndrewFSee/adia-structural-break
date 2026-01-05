"""
Training with TabPFN - used by 10th place for +4% boost.

TabPFN is a transformer-based model pretrained on synthetic tabular data.
It excels on small datasets (<10k samples, <100 features) without hyperparameter tuning.

Strategy:
1. Use top N features from feature importance (not RFE - too slow)
2. Use TabPFN for prediction
3. Optionally blend with LightGBM

10th place: 85.86 AUC → +4% with TabPFN
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from sb import data_loader, features, config

print("="*70)
print("TABPFN + FEATURE SELECTION (10TH PLACE APPROACH)")
print("="*70)
print()

# Check if TabPFN is available
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    print("✓ TabPFN is available")
except ImportError:
    TABPFN_AVAILABLE = False
    print("✗ TabPFN not installed. Install with: pip install tabpfn")
    print("  Falling back to LightGBM-only approach with feature selection")

# Load data
print("\nLoading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(df['id'].unique())} series, break rate: {y.mean()*100:.2f}%")

# Extract Phase 1 features
print("\nExtracting Phase 1 features...")
X_phase1 = features.base.compute_features(
    df,
    use_multiscale=True,
    use_cv=True,
    use_transforms=True,
    use_compression=True,
    use_cusum=True,
    use_boundary_dist=True,
    use_boundary_tail_shape=True
)
print(f"Phase 1 features: {X_phase1.shape}")

# Impute NaN
X_all = X_phase1.fillna(X_phase1.median())

# Remove any duplicate columns
X_all = X_all.loc[:, ~X_all.columns.duplicated()]
print(f"After deduplication: {X_all.shape}")

# Step 1: Get feature importance from LightGBM to select top features
print("\n" + "="*70)
print("STEP 1: GET FEATURE IMPORTANCE FROM LIGHTGBM")
print("="*70 + "\n")

# Train quick LightGBM to get feature importances
X_ranked = X_all.rank(pct=True)
train_data = lgb.Dataset(X_ranked, label=y)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
}

# Train model
model = lgb.train(params, train_data, num_boost_round=200)

# Get feature importances
importance = pd.DataFrame({
    'feature': X_all.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("Top 20 features by importance:")
print(importance.head(20).to_string(index=False))

# Step 2: Test different feature counts
print("\n" + "="*70)
print("STEP 2: TEST DIFFERENT FEATURE COUNTS")
print("="*70 + "\n")

feature_counts = [33, 50, 75, 100, 150]
results = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_features in feature_counts:
    top_features = importance['feature'].head(n_features).tolist()
    X_subset = X_all[top_features]
    
    fold_aucs_lgb = []
    fold_aucs_tabpfn = []
    fold_aucs_blend = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_subset, y), 1):
        X_train = X_subset.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_subset.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Rank normalize
        X_train_ranked = X_train.rank(pct=True)
        X_val_ranked = X_val.rank(pct=True)
        
        # LightGBM
        train_data = lgb.Dataset(X_train_ranked, label=y_train)
        model_lgb = lgb.train(params, train_data, num_boost_round=200)
        pred_lgb = model_lgb.predict(X_val_ranked)
        auc_lgb = roc_auc_score(y_val, pred_lgb)
        fold_aucs_lgb.append(auc_lgb)
        
        # TabPFN (if available)
        if TABPFN_AVAILABLE and n_features <= 100:  # TabPFN has feature limit
            try:
                model_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
                model_tabpfn.fit(X_train_ranked.values, y_train.values)
                pred_tabpfn = model_tabpfn.predict_proba(X_val_ranked.values)[:, 1]
                auc_tabpfn = roc_auc_score(y_val, pred_tabpfn)
                fold_aucs_tabpfn.append(auc_tabpfn)
                
                # Blend (simple average)
                pred_blend = (pred_lgb + pred_tabpfn) / 2
                auc_blend = roc_auc_score(y_val, pred_blend)
                fold_aucs_blend.append(auc_blend)
            except Exception as e:
                print(f"  TabPFN error in fold {fold_idx}: {e}")
    
    mean_lgb = np.mean(fold_aucs_lgb)
    std_lgb = np.std(fold_aucs_lgb)
    
    result = {
        'n_features': n_features,
        'lgb_auc': mean_lgb,
        'lgb_std': std_lgb,
    }
    
    if fold_aucs_tabpfn:
        mean_tabpfn = np.mean(fold_aucs_tabpfn)
        std_tabpfn = np.std(fold_aucs_tabpfn)
        mean_blend = np.mean(fold_aucs_blend)
        std_blend = np.std(fold_aucs_blend)
        result['tabpfn_auc'] = mean_tabpfn
        result['tabpfn_std'] = std_tabpfn
        result['blend_auc'] = mean_blend
        result['blend_std'] = std_blend
        print(f"n={n_features:3d}: LGB={mean_lgb:.4f}±{std_lgb:.4f}, TabPFN={mean_tabpfn:.4f}±{std_tabpfn:.4f}, Blend={mean_blend:.4f}±{std_blend:.4f}")
    else:
        print(f"n={n_features:3d}: LGB={mean_lgb:.4f}±{std_lgb:.4f}")
    
    results.append(result)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70 + "\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best configuration
if 'blend_auc' in results_df.columns:
    best_idx = results_df['blend_auc'].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\nBest: n_features={int(best['n_features'])}, Blend AUC={best['blend_auc']:.4f}")
else:
    best_idx = results_df['lgb_auc'].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\nBest: n_features={int(best['n_features'])}, LGB AUC={best['lgb_auc']:.4f}")

print("\nTo install TabPFN: pip install tabpfn")
print("TabPFN typically gives +2-4% boost on small tabular datasets")

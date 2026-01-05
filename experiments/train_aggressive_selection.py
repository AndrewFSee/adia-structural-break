"""
Ultra-aggressive feature selection (10th place: 33 features → 85.86 AUC)

Test multiple feature counts: 33, 50, 75, 100, 150
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from sb import data_loader, features

print("="*70)
print("ULTRA-AGGRESSIVE FEATURE SELECTION (10TH PLACE: 33 FEATURES)")
print("="*70)
print()

# Load data
print("Loading data...")
df, y = data_loader.load_for_training("data")
print(f"Loaded {len(df['id'].unique())} series, break rate: {y.mean()*100:.2f}%\n")

# Extract features
print("Extracting features...")
X_phase1 = features.base.compute_features(
    df, use_multiscale=True, use_cv=True, use_transforms=True,
    use_compression=True, use_cusum=True, use_boundary_dist=True,
    use_boundary_tail_shape=True
)
from sb.features.statistical_tests import compute_statistical_test_features
X_stats = compute_statistical_test_features(
    df, use_anderson=True, use_cohens_d=True, use_variance_ratios=True,
    use_iqr_ratios=True, use_hypothesis_tests=True, use_rolling_stats=True
)
X_all = pd.concat([X_phase1, X_stats], axis=1)
X_all = X_all.fillna(X_all.median()).loc[:, ~X_all.columns.duplicated()]
print(f"Total features: {X_all.shape}\n")

# Get feature importance
X_ranked = X_all.rank(pct=True)
train_data = lgb.Dataset(X_ranked, label=y)
params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.03, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 20,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'seed': 42,
}
model = lgb.train(params, train_data, num_boost_round=300)
importance = pd.DataFrame({
    'feature': X_all.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

# Test multiple feature counts
feature_counts = [33, 50, 75, 100, 150]
results = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_feat in feature_counts:
    print(f"\n{'='*70}")
    print(f"TESTING {n_feat} FEATURES")
    print('='*70)
    
    top_features = importance['feature'].head(n_feat).tolist()
    X_sel = X_all[top_features]
    
    # 5 diverse models
    oof_models = [np.zeros(len(y)) for _ in range(5)]
    fold_aucs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_sel, y), 1):
        X_train = X_sel.iloc[train_idx].rank(pct=True)
        X_val = X_sel.iloc[val_idx].rank(pct=True)
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        fold_preds = []
        for i, seed in enumerate([42, 123, 456, 789, 999]):
            p = params.copy()
            p['seed'] = seed
            if i == 1: p['num_leaves'], p['reg_alpha'], p['reg_lambda'] = 127, 0.3, 0.3
            elif i == 2: p['num_leaves'], p['reg_alpha'], p['reg_lambda'] = 31, 0.0, 0.0
            elif i == 3: p['feature_fraction'], p['bagging_fraction'] = 0.9, 0.9
            elif i == 4: p['learning_rate'] = 0.01
            
            train_d = lgb.Dataset(X_train, label=y_train)
            val_d = lgb.Dataset(X_val, label=y_val, reference=train_d)
            m = lgb.train(p, train_d, num_boost_round=800 if i==4 else 500,
                         valid_sets=[val_d], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            pred = m.predict(X_val)
            oof_models[i][val_idx] = pred
            fold_preds.append(pred)
        
        pred_avg = np.mean(fold_preds, axis=0)
        auc = roc_auc_score(y_val, pred_avg)
        fold_aucs.append(auc)
        print(f"  Fold {fold_idx}: {auc:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    # Calibration
    ensemble = np.mean(oof_models, axis=0)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(ensemble, y)
    cal_preds = iso.predict(ensemble)
    
    cal_aucs = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_sel, y), 1):
        auc = roc_auc_score(y.iloc[val_idx], cal_preds[val_idx])
        cal_aucs.append(auc)
    
    cal_mean = np.mean(cal_aucs)
    cal_std = np.std(cal_aucs)
    
    results.append({
        'n_features': n_feat,
        'ensemble_auc': mean_auc,
        'ensemble_std': std_auc,
        'calibrated_auc': cal_mean,
        'calibrated_std': cal_std,
    })
    
    print(f"\n  Ensemble:   {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Calibrated: {cal_mean:.4f} ± {cal_std:.4f}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print('='*70)
print()

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_idx = results_df['calibrated_auc'].idxmax()
best = results_df.iloc[best_idx]
print(f"\n🏆 BEST: {int(best['n_features'])} features → {best['calibrated_auc']:.4f} AUC")

# Show top features for best configuration
best_n = int(best['n_features'])
print(f"\nTop {min(20, best_n)} features:")
print(importance.head(min(20, best_n)).to_string(index=False))

"""
Model stacking approach - used by Brazilian team (1st place).

Strategy:
1. Train multiple diverse models (LightGBM, XGBoost, CatBoost)
2. Stack their predictions
3. Use meta-learner for final prediction

Also includes:
- Feature selection by importance
- Hyperparameter tuning for each base model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from sb import data_loader, features, config

print("="*70)
print("MODEL STACKING (BRAZILIAN TEAM 1ST PLACE APPROACH)")
print("="*70)
print()

# Check for optional libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✓ XGBoost available")
except ImportError:
    XGB_AVAILABLE = False
    print("✗ XGBoost not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    print("✓ CatBoost available")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("✗ CatBoost not available")

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

# Add statistical test features (used by winners)
print("Adding statistical test features...")
from sb.features.statistical_tests import compute_statistical_test_features
X_stats = compute_statistical_test_features(
    df,
    use_anderson=True,
    use_cohens_d=True,
    use_variance_ratios=True,
    use_iqr_ratios=True,
    use_hypothesis_tests=True,
    use_rolling_stats=True,
)
print(f"Statistical test features: {X_stats.shape}")

# Combine all features
X_all = pd.concat([X_phase1, X_stats], axis=1)

# Impute NaN and remove duplicates
X_all = X_all.fillna(X_all.median())
X_all = X_all.loc[:, ~X_all.columns.duplicated()]
print(f"After cleanup: {X_all.shape}")

# Get feature importance and select top features
print("\n" + "="*70)
print("STEP 1: FEATURE SELECTION BY IMPORTANCE")
print("="*70 + "\n")

X_ranked = X_all.rank(pct=True)
train_data = lgb.Dataset(X_ranked, label=y)

params_lgb = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,  # Increased from 31
    'learning_rate': 0.03,  # Reduced for better convergence
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
    'verbose': -1,
    'seed': 42,
}

model = lgb.train(params_lgb, train_data, num_boost_round=200)
importance = pd.DataFrame({
    'feature': X_all.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

# Select top 100 features (sweet spot for stacking)
N_FEATURES = 150  # Try more features with statistical tests included
top_features = importance['feature'].head(N_FEATURES).tolist()
X_selected = X_all[top_features]
print(f"Selected top {N_FEATURES} features")
print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))

# Stacking with CV
print("\n" + "="*70)
print("STEP 2: STACKING WITH 5-FOLD CV")
print("="*70 + "\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store OOF predictions for stacking
oof_lgb = np.zeros(len(y))
oof_xgb = np.zeros(len(y)) if XGB_AVAILABLE else None
oof_cat = np.zeros(len(y)) if CATBOOST_AVAILABLE else None

fold_aucs_lgb = []
fold_aucs_xgb = []
fold_aucs_cat = []
fold_aucs_stack = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
    print(f"\n--- Fold {fold_idx}/5 ---")
    
    X_train = X_selected.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X_selected.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    # Rank normalize
    X_train_ranked = X_train.rank(pct=True)
    X_val_ranked = X_val.rank(pct=True)
    
    # LightGBM
    train_data = lgb.Dataset(X_train_ranked, label=y_train)
    val_data = lgb.Dataset(X_val_ranked, label=y_val, reference=train_data)
    
    model_lgb = lgb.train(
        params_lgb, 
        train_data, 
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    pred_lgb = model_lgb.predict(X_val_ranked)
    oof_lgb[val_idx] = pred_lgb
    auc_lgb = roc_auc_score(y_val, pred_lgb)
    fold_aucs_lgb.append(auc_lgb)
    print(f"  LightGBM: {auc_lgb:.4f}")
    
    # XGBoost
    if XGB_AVAILABLE:
        params_xgb = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,  # Increased from 5
            'learning_rate': 0.03,  # Reduced for better convergence
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'seed': 42,
            'verbosity': 0,
        }
        
        dtrain = xgb.DMatrix(X_train_ranked, label=y_train)
        dval = xgb.DMatrix(X_val_ranked, label=y_val)
        
        model_xgb = xgb.train(
            params_xgb,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        pred_xgb = model_xgb.predict(dval)
        oof_xgb[val_idx] = pred_xgb
        auc_xgb = roc_auc_score(y_val, pred_xgb)
        fold_aucs_xgb.append(auc_xgb)
        print(f"  XGBoost:  {auc_xgb:.4f}")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        model_cat = CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,  # Reduced
            depth=6,  # Increased from 5
            l2_leaf_reg=3,  # Regularization
            verbose=False,
            random_state=42,
            early_stopping_rounds=50
        )
        model_cat.fit(X_train_ranked, y_train, eval_set=(X_val_ranked, y_val))
        pred_cat = model_cat.predict_proba(X_val_ranked)[:, 1]
        oof_cat[val_idx] = pred_cat
        auc_cat = roc_auc_score(y_val, pred_cat)
        fold_aucs_cat.append(auc_cat)
        print(f"  CatBoost: {auc_cat:.4f}")
    
    # Simple average stacking
    preds = [pred_lgb]
    if XGB_AVAILABLE:
        preds.append(pred_xgb)
    if CATBOOST_AVAILABLE:
        preds.append(pred_cat)
    
    pred_stack = np.mean(preds, axis=0)
    auc_stack = roc_auc_score(y_val, pred_stack)
    fold_aucs_stack.append(auc_stack)
    print(f"  Stack (avg): {auc_stack:.4f}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70 + "\n")

print(f"LightGBM:     {np.mean(fold_aucs_lgb):.4f} ± {np.std(fold_aucs_lgb):.4f}")
if XGB_AVAILABLE:
    print(f"XGBoost:      {np.mean(fold_aucs_xgb):.4f} ± {np.std(fold_aucs_xgb):.4f}")
if CATBOOST_AVAILABLE:
    print(f"CatBoost:     {np.mean(fold_aucs_cat):.4f} ± {np.std(fold_aucs_cat):.4f}")
print(f"Stack (avg):  {np.mean(fold_aucs_stack):.4f} ± {np.std(fold_aucs_stack):.4f}")

# Try weighted stacking
print("\n--- Optimizing Stack Weights ---")
best_auc = 0
best_weights = None

# Grid search over weights
for w1 in np.arange(0.2, 0.8, 0.1):
    if XGB_AVAILABLE and CATBOOST_AVAILABLE:
        for w2 in np.arange(0.1, 0.5, 0.1):
            w3 = 1 - w1 - w2
            if w3 > 0:
                pred_weighted = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
                auc = roc_auc_score(y, pred_weighted)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = (w1, w2, w3)
    elif XGB_AVAILABLE:
        w2 = 1 - w1
        pred_weighted = w1 * oof_lgb + w2 * oof_xgb
        auc = roc_auc_score(y, pred_weighted)
        if auc > best_auc:
            best_auc = auc
            best_weights = (w1, w2)
    else:
        # Only LightGBM
        best_auc = roc_auc_score(y, oof_lgb)
        best_weights = (1.0,)
        break

print(f"\nBest weighted ensemble: {best_auc:.4f}")
if best_weights:
    print(f"Weights: {best_weights}")

# Meta-learner stacking
print("\n--- Meta-Learner Stacking ---")
stack_features = [oof_lgb]
if XGB_AVAILABLE:
    stack_features.append(oof_xgb)
if CATBOOST_AVAILABLE:
    stack_features.append(oof_cat)

X_stack = np.column_stack(stack_features)

# Train logistic regression as meta-learner with CV
meta_aucs = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_stack, y), 1):
    X_train_meta = X_stack[train_idx]
    y_train_meta = y.iloc[train_idx]
    X_val_meta = X_stack[val_idx]
    y_val_meta = y.iloc[val_idx]
    
    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(X_train_meta, y_train_meta)
    pred_meta = meta_model.predict_proba(X_val_meta)[:, 1]
    auc_meta = roc_auc_score(y_val_meta, pred_meta)
    meta_aucs.append(auc_meta)

print(f"Meta-learner (LogReg): {np.mean(meta_aucs):.4f} ± {np.std(meta_aucs):.4f}")

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\nBest single model (LightGBM): {np.mean(fold_aucs_lgb):.4f}")
print(f"Simple average stacking:      {np.mean(fold_aucs_stack):.4f}")
print(f"Weighted stacking:            {best_auc:.4f}")
print(f"Meta-learner stacking:        {np.mean(meta_aucs):.4f}")

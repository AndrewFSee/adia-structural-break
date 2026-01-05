"""
Test autoencoder features with PyTorch (now working!).

Uses a simple autoencoder to learn compressed representations of the
time series data and extract features from the latent space.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score

from sb import data_loader
from sb import features
from sb.features.statistical_tests import compute_statistical_test_features


def main():
    print("Loading data...")
    df_train, y_train = data_loader.load_for_training("data")
    print(f"Loaded {len(y_train)} series, break rate: {y_train.mean()*100:.2f}%\n")
    
    print("Extracting base features...")
    X_base = features.base.compute_features(
        df_train,
        use_multiscale=True,
        use_cv=True,
        use_transforms=True,
        use_compression=True,
        use_cusum=True,
        use_boundary_dist=True,
        use_boundary_tail_shape=True
    )
    print(f"Base features: {X_base.shape}")
    
    print("\nExtracting statistical test features...")
    X_stat = compute_statistical_test_features(
        df_train,
        use_anderson=True,
        use_cohens_d=True,
        use_variance_ratios=True,
        use_iqr_ratios=True,
        use_hypothesis_tests=True,
        use_rolling_stats=True,
    )
    print(f"Statistical features: {X_stat.shape}")
    
    print("\nExtracting advanced test features...")
    sys.path.insert(0, str(Path(__file__).parent / "src" / "sb" / "features"))
    import advanced_tests
    import importlib
    importlib.reload(advanced_tests)
    X_adv = advanced_tests.extract_features(df_train)
    print(f"Advanced features: {X_adv.shape}")
    
    print("\nExtracting autoencoder features (PyTorch)...")
    import autoencoder
    importlib.reload(autoencoder)
    
    # Train autoencoder with reasonable settings
    X_ae = autoencoder.compute_autoencoder_features(
        df_train,
        window_size=20,
        stride=5,
        epochs=50,
        verbose=True
    )
    print(f"Autoencoder features: {X_ae.shape}")
    
    # Combine all features
    X = pd.concat([X_base, X_stat, X_adv, X_ae], axis=1)
    print(f"\nTotal features: {X.shape}")
    
    # Remove duplicates and handle NaN
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Select top 100 features
    print("\nSelecting top 100 features...")
    selector = SelectKBest(mutual_info_classif, k=100)
    X_selected = selector.fit_transform(X, y_train)
    
    # Count how many autoencoder features were selected
    feature_names = X.columns
    selected_mask = selector.get_support()
    selected_features = feature_names[selected_mask]
    
    ae_features_selected = [f for f in selected_features if f.startswith('ae_')]
    print(f"Autoencoder features in top 100: {len(ae_features_selected)}")
    
    if ae_features_selected:
        print("\nTop autoencoder features:")
        # Get feature importances
        lgbm_temp = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        lgbm_temp.fit(X_selected, y_train)
        
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': lgbm_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        ae_importances = feature_importance[feature_importance['feature'].isin(ae_features_selected)]
        for _, row in ae_importances.head(10).iterrows():
            rank = feature_importance[feature_importance['feature'] == row['feature']].index[0] + 1
            print(f"  {row['feature']} (#{rank} overall: {row['importance']:.2f})")
    
    # Train diverse ensemble
    print("\nTraining diverse ensemble models...")
    
    configs = [
        {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42, 'verbose': -1},
        {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.03, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 123, 'verbose': -1},
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': 456, 'verbose': -1},
        {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 789, 'verbose': -1},
        {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.02, 'random_state': 101, 'verbose': -1},
    ]
    
    print("\nIndividual Models:")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_aucs = []
    
    for train_idx, val_idx in kf.split(X_selected, y_train):
        X_train_fold = X_selected[train_idx]
        X_val_fold = X_selected[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        fold_preds = []
        for i, config in enumerate(configs, 1):
            model = LGBMClassifier(**config)
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict_proba(X_val_fold)[:, 1]
            fold_preds.append(preds)
            
        fold_ensemble = np.mean(fold_preds, axis=0)
        fold_auc = roc_auc_score(y_val_fold, fold_ensemble)
        ensemble_aucs.append(fold_auc)
    
    # Also show individual model performance
    for i, config in enumerate(configs, 1):
        model = LGBMClassifier(**config)
        scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"Model {i}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    print(f"\nEnsemble: {np.mean(ensemble_aucs):.4f} AUC")
    
    # Compare to baseline (0.8966)
    baseline = 0.8966
    improvement = np.mean(ensemble_aucs) - baseline
    
    if improvement > 0:
        print(f"\n✅ Improvement: +{improvement:.4f} AUC")
    else:
        print(f"\n❌ No improvement: {improvement:.4f} AUC")


if __name__ == '__main__':
    main()

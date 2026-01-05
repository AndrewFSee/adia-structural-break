# Example Usage

## Basic Usage Examples

### Example 1: Quick Test with Synthetic Data

```python
import numpy as np
import pandas as pd
from solution import train, infer

# Generate synthetic time series
np.random.seed(42)
data = []

for i in range(10):
    series_id = f"series_{i:03d}"
    
    # Pre-break: normal distribution
    x0 = np.random.randn(200)
    
    # Post-break: shift mean for half the series
    if i < 5:
        x1 = np.random.randn(200)  # No break
    else:
        x1 = np.random.randn(200) * 1.5 + 1.0  # Break!
    
    # Format as required
    for t, val in enumerate(x0):
        data.append({"id": series_id, "period": 0, "time": t, "value": val})
    for t, val in enumerate(x1):
        data.append({"id": series_id, "period": 1, "time": t, "value": val})

df = pd.DataFrame(data)

# Run inference
predictions = infer(df)
print(predictions)

# Expected: series_005 through series_009 should have higher scores
```

### Example 2: Load Real Data and Train

```python
import pandas as pd
from sb import io, pipeline

# Load your training data
df_train = pd.read_csv("train.csv")

# If labels are in separate file
y_train = pd.read_csv("train_labels.csv").set_index("id")["label"]

# Run baseline pipeline (no ML)
scores = pipeline.run_baseline_pipeline(df_train, y_train)

# Output shows:
# - Feature computation time
# - ROC AUC score
# - Score distribution
# - Label distribution
```

### Example 3: Extract and Inspect Features

```python
from sb.features import base
import pandas as pd

# Load data
df = pd.read_csv("train.csv")

# Compute features for all series
features = base.compute_features(df)

# Inspect
print("Feature columns:", features.columns.tolist())
print("\nFeature statistics:")
print(features.describe())

# Check correlations
print("\nFeature correlations:")
print(features.corr())

# Save for later analysis
features.to_csv("extracted_features.csv")
```

### Example 4: Manual Feature Computation

```python
import numpy as np
from sb import io, preprocessing
from sb.features import dist, dynamics

# Load data for one series
df = pd.read_csv("train.csv")
df_single = df[df["id"] == "series_001"]

# Split into pre/post
x0, x1 = io.split_series(df_single)

# Apply robust scaling
x0_scaled = preprocessing.robust_scale(x0)
x1_scaled = preprocessing.robust_scale(x1)

# Compute individual features
quant_feats = dist.quantile_features(x0_scaled, x1_scaled)
entropy_feat = dist.entropy_change(x0_scaled, x1_scaled)
vol_feats = dynamics.volatility_features(x0_scaled, x1_scaled)

print("Quantile features:", quant_feats)
print("Entropy change:", entropy_feat)
print("Volatility features:", vol_feats)
```

### Example 5: Cross-Validation

```python
from sb import cv, features
import pandas as pd

# Load data
df = pd.read_csv("train.csv")
y = pd.read_csv("train_labels.csv").set_index("id")["label"]

# Compute features
X = features.base.compute_features(df)

# Split for validation
X_train, X_test, y_train, y_test = cv.stratified_split_by_id(X, y)

print(f"Training set: {len(X_train)} series")
print(f"Test set: {len(X_test)} series")

# Evaluate baseline on test set
from sb.features.base import aggregate_features
test_scores = aggregate_features(X_test)

# Compute test AUC
from sklearn.metrics import roc_auc_score
test_auc = roc_auc_score(y_test, test_scores)
print(f"Test ROC AUC: {test_auc:.4f}")
```

### Example 6: Using LightGBM Meta-Model

```python
from sb import features, models
import pandas as pd

# Load data
df_train = pd.read_csv("train.csv")
y_train = pd.read_csv("train_labels.csv").set_index("id")["label"]

# Compute features
X_train = features.base.compute_features(df_train)

# Train LightGBM
model = models.gbm.train_gbm(X_train, y_train)

# Check feature importance
importance = model.get_feature_importance()
print("Top features:")
print(importance.head())

# Predict on new data
df_test = pd.read_csv("test.csv")
X_test = features.base.compute_features(df_test)
predictions = model.predict(X_test)
```

### Example 7: Visualize Score Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt
from solution import infer

# Load data
df = pd.read_csv("train.csv")
y = pd.read_csv("train_labels.csv").set_index("id")["label"]

# Get predictions
scores = infer(df)

# Align with labels
scores_aligned = scores.loc[y.index]

# Plot distribution by label
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data=pd.DataFrame({
    'score': scores_aligned,
    'label': y
}), x='score', hue='label', bins=30, stat='density', common_norm=False)
plt.xlabel('Predicted Score')
plt.ylabel('Density')
plt.title('Score Distribution by Label')
plt.legend(['No Break (0)', 'Break (1)'])
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=150)
plt.show()

# Good separation = high AUC
```

### Example 8: Batch Inference for Submission

```python
from solution import infer
from sb import io
import pandas as pd

# Load test data (no labels)
df_test = pd.read_csv("test.csv")

print(f"Loaded {df_test['id'].nunique()} test series")

# Run inference
predictions = infer(df_test)

# Save in submission format
io.save_predictions(predictions, "submission.csv")

print(f"Saved predictions to submission.csv")
print(f"Score range: [{predictions.min():.4f}, {predictions.max():.4f}]")

# Verify submission format
submission = pd.read_csv("submission.csv")
print("\nSubmission preview:")
print(submission.head())
```

### Example 9: Compare Baseline vs GBM

```python
from sb import features, models, cv
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv("train.csv")
y = pd.read_csv("train_labels.csv").set_index("id")["label"]

# Compute features
X = features.base.compute_features(df)

# Split
X_train, X_test, y_train, y_test = cv.stratified_split_by_id(X, y, test_size=0.3)

# Method 1: Baseline (rank aggregation)
baseline_scores = features.base.aggregate_features(X_test)
baseline_auc = roc_auc_score(y_test, baseline_scores)

# Method 2: LightGBM
model = models.gbm.train_gbm(X_train, y_train)
gbm_scores = model.predict(X_test)
gbm_auc = roc_auc_score(y_test, gbm_scores)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"Baseline (rank aggregation): {baseline_auc:.4f}")
print(f"LightGBM meta-model:         {gbm_auc:.4f}")
print(f"Improvement:                 {(gbm_auc - baseline_auc):.4f}")
print("=" * 60)

# Decide which to use for submission
```

### Example 10: Debugging a Single Series

```python
from sb import io, preprocessing, features
import pandas as pd
import matplotlib.pyplot as plt

# Load data for one specific series
df = pd.read_csv("train.csv")
series_id = "series_001"
df_single = df[df["id"] == series_id]

# Get label
y = pd.read_csv("train_labels.csv").set_index("id")
label = y.loc[series_id, "label"]

# Split
x0, x1 = io.split_series(df_single)

# Plot raw series
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(x0, alpha=0.7, label='Period 0 (pre-break)')
axes[0].plot(len(x0) + np.arange(len(x1)), x1, alpha=0.7, label='Period 1 (post-break)')
axes[0].axvline(len(x0), color='red', linestyle='--', label='Break point')
axes[0].set_title(f'{series_id} - Label: {label}')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Apply scaling
x0_scaled = preprocessing.robust_scale(x0)
x1_scaled = preprocessing.robust_scale(x1)

# Plot scaled
axes[1].plot(x0_scaled, alpha=0.7, label='Period 0 (scaled)')
axes[1].plot(len(x0) + np.arange(len(x1)), x1_scaled, alpha=0.7, label='Period 1 (scaled)')
axes[1].axvline(len(x0), color='red', linestyle='--', label='Break point')
axes[1].set_title('After Robust Scaling')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Scaled Value')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{series_id}_debug.png', dpi=150)
plt.show()

# Compute features
feats = features.base.compute_single_series_features(x0, x1)
print(f"\nFeatures for {series_id}:")
for k, v in feats.items():
    print(f"  {k:20s}: {v:.6f}")

# Get final score
all_features = features.base.compute_features(df)
score = features.base.aggregate_features(all_features).loc[series_id]
print(f"\nFinal score: {score:.4f}")
print(f"True label:  {label}")
```

## Command-Line Usage

### Quick sanity check
```bash
python scripts/sanity_check.py
```

### Train on your data
```bash
# Basic (with labels in data file)
python scripts/train_local.py --data train.csv

# With separate label file
python scripts/train_local.py --data train.csv --labels labels.csv

# With LightGBM
python scripts/train_local.py --data train.csv --mode gbm
```

### Generate predictions
```bash
# Basic inference
python scripts/infer_local.py --data test.csv --output submission.csv

# With evaluation (if you have test labels)
python scripts/infer_local.py --data test.csv --labels test_labels.csv --output predictions.csv
```

### Extract features for analysis
```bash
python scripts/make_features.py --data train.csv --output features.csv
```

### Run tests
```bash
# Feature tests
python tests/test_features.py

# Determinism tests
python tests/test_determinism.py
```

## Integration with CrunchDAO Platform

The platform will call your functions like this:

```python
# During training phase (may be called once or not at all)
train(X_train, y_train)

# During inference phase (called on test data)
predictions = infer(X_test)
```

Your `solution.py` must:
1. Define these two functions
2. Return predictions as `pd.Series` with index=id
3. Ensure predictions are in [0, 1]
4. Be deterministic (same input → same output)

---

See [QUICKSTART.md](QUICKSTART.md) for more detailed usage instructions.

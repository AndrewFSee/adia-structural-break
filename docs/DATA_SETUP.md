# Working with CrunchDAO Data

Your data is now set up in the correct format!

## Data Location

```
data/
├── X_train.parquet          (10,000 time series)
├── y_train.parquet          (labels for training)
├── X_test.reduced.parquet   (100 time series)
└── y_test.reduced.parquet   (labels for testing)
```

## Quick Commands

### 1. Inspect the Data
```bash
python scripts/inspect_data.py
```
Shows dataset statistics, sample series, and data format.

### 2. Train on CrunchDAO Data
```bash
# Using CrunchDAO format (automatic)
python scripts/train_local.py

# Or explicitly
python scripts/train_local.py --crunchdao
```

### 3. Test on CrunchDAO Data
```bash
# Run inference on test set
python scripts/infer_local.py

# Or explicitly
python scripts/infer_local.py --crunchdao --output predictions.csv
```

### 4. Sanity Checks
```bash
python scripts/sanity_check.py
```

## Data Format

### X (Features)
- **Format**: MultiIndex DataFrame with (id, time)
- **Columns**:
  - `value`: The time series values
  - `period`: 0=before break point, 1=after break point

### y (Labels)
- **Format**: Series with index=id
- **Values**: `structural_breakpoint` (bool)
  - `True` (1) = structural break occurred
  - `False` (0) = no structural break

## Examples

### Load and Inspect Training Data
```python
from sb import data_loader

# Load in CrunchDAO format
X_train, y_train = data_loader.load_crunchdao_data("data", "train")
print(f"Training series: {len(X_train.index.get_level_values('id').unique())}")
print(f"Break rate: {y_train.mean():.1%}")

# Or load in standard format (flat DataFrame)
df_train, y_train = data_loader.load_for_training("data")
print(df_train.head())
```

### Get Dataset Statistics
```python
from sb import data_loader

stats = data_loader.get_dataset_stats("data")
print(f"Training set: {stats['train']['n_series']} series")
print(f"Test set: {stats['test']['n_series']} series")
```

### Run Full Training Pipeline
```python
from sb import data_loader, pipeline

# Load data
df_train, y_train = data_loader.load_for_training("data")

# Run baseline
scores = pipeline.run_baseline_pipeline(df_train, y_train)
```

## What the Scripts Do

### `inspect_data.py`
- Shows dataset statistics (size, break rate, etc.)
- Displays sample series data
- Helps you understand the data structure

### `train_local.py`
- Loads training data
- Computes features
- Evaluates model performance (ROC AUC)
- Shows score distributions

### `infer_local.py`
- Loads test data
- Generates predictions
- Optionally evaluates against test labels
- Saves predictions to CSV

## Tips

1. **Always start with `inspect_data.py`** to understand your data
2. **Use `--crunchdao` flag** or just omit `--data` to auto-load from data/
3. **Test set labels are available** for local evaluation
4. **The data loader handles format conversion** automatically

## Expected Performance

With 10,000 training series and the Day 1-2 baseline:
- **Target ROC AUC**: 0.80+
- **Processing time**: ~10-30 seconds for full dataset
- **Memory usage**: ~500MB-1GB

## Next Steps

1. Run `python scripts/inspect_data.py` to see your data
2. Run `python scripts/train_local.py` to evaluate baseline
3. Check the ROC AUC score
4. If good (0.80+), test on test set with `python scripts/infer_local.py`
5. Submit to CrunchDAO platform!

# PyTorch DLL Fix Guide

## Problem
PyTorch DLL error on Windows: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

**UPDATE**: Getting ClobberError when using conda (pip/conda package manager conflict)

## Quick Fix (Works Around ClobberError)

```powershell
# Clean uninstall with pip (manages what pip installed)
pip uninstall -y torch torchvision torchaudio

# Install CPU-only with pip (avoids conda conflict)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Test it works
python -c "import torch; print('PyTorch', torch.__version__, 'works!')"
```

## Original Fix 1 (Causes ClobberError - Don't Use)
~~conda install pytorch cpuonly -c pytorch~~ ← This conflicts with pip-installed packages

### Fix 2: Use pip CPU version
```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Fix 3: Install Visual C++ Redistributables
Download and install: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Fix 4: Update conda environment
```powershell
conda update conda
conda update --all
# Then try Fix 1 again
```

## Test Installation
```powershell
python -c "import torch; print('PyTorch', torch.__version__, 'works!')"
```

## Alternative: Skip Autoencoder
The **advanced statistical tests** are running now and likely to provide better results anyway:
- Cramér-von Mises test
- Energy statistic
- Wilcoxon rank-sum
- Mood's test
- Tail features
- Spectral features
- Permutation entropy

These are proven effective for structural breaks and don't require PyTorch.

## Current Status
✅ **train_advanced_tests.py** - Running (testing 20+ new features)
❌ **train_with_autoencoder.py** - Blocked by PyTorch DLL issue

## Recommendation
1. Wait for advanced_tests results (likely to improve score)
2. If needed, try PyTorch Fix 1 above
3. Focus on winning approaches that don't require PyTorch

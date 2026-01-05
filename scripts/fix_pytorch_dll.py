"""
Troubleshoot and fix PyTorch DLL issues on Windows.

Common fixes:
1. Reinstall PyTorch with CPU-only version (more stable on Windows)
2. Install Visual C++ Redistributables
3. Update conda/pip
4. Clean install PyTorch
"""

import sys
import subprocess
import os

print("="*70)
print("PYTORCH DLL ISSUE TROUBLESHOOTER")
print("="*70)
print()

# Test current PyTorch installation
print("1. Testing current PyTorch installation...")
try:
    import torch
    print(f"   ✅ PyTorch imported successfully!")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print()
    print("   PyTorch is working! No fixes needed.")
    sys.exit(0)
except Exception as e:
    print(f"   ❌ Error: {e}")
    print()

# Fix attempts
print("2. Attempting fixes...")
print()

print("   Option A: Reinstall PyTorch (CPU-only, most stable)")
print("   Command: conda install pytorch cpuonly -c pytorch")
print()

print("   Option B: Use pip version")
print("   Command: pip uninstall torch; pip install torch --index-url https://download.pytorch.org/whl/cpu")
print()

print("   Option C: Install Visual C++ Redistributables")
print("   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
print()

response = input("Try Option A (conda CPU-only reinstall)? (y/n): ")

if response.lower() == 'y':
    print("\nReinstalling PyTorch (CPU-only)...")
    print("This may take a few minutes...")
    
    # Uninstall current version
    print("\n[1/2] Uninstalling current PyTorch...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                   capture_output=False)
    
    # Install CPU-only version via conda
    print("\n[2/2] Installing CPU-only PyTorch via conda...")
    subprocess.run(["conda", "install", "-y", "pytorch", "cpuonly", "-c", "pytorch"],
                   capture_output=False, shell=True)
    
    print("\n" + "="*70)
    print("Installation complete! Testing...")
    print("="*70)
    print()
    
    try:
        import torch
        print(f"✅ SUCCESS! PyTorch {torch.__version__} is now working!")
        print()
        print("You can now run:")
        print("  python train_with_autoencoder.py")
    except Exception as e:
        print(f"❌ Still having issues: {e}")
        print()
        print("Try Option C: Download and install Visual C++ Redistributables")
        print("https://aka.ms/vs/17/release/vc_redist.x64.exe")
else:
    print("\nSkipping automatic fix.")
    print()
    print("Manual fix options:")
    print()
    print("1. CPU-only PyTorch (recommended):")
    print("   conda install pytorch cpuonly -c pytorch")
    print()
    print("2. Or use pip:")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("3. Install Visual C++ Redistributables:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")

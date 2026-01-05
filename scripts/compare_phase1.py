"""
Quick comparison script to test Phase 1 features impact.

Runs multiple configurations and compares AUC scores.
"""

import subprocess
import re

configs = [
    ("Baseline", ["--multiscale"]),
    ("+ CV", ["--multiscale", "--cv"]),
    ("+ CV + Transforms", ["--multiscale", "--cv", "--transforms"]),
    ("Full Phase 1", ["--multiscale", "--phase1"]),
]

print("="*70)
print("PHASE 1 FEATURE COMPARISON")
print("="*70)
print("\nThis will run 4 training configurations and compare AUC scores.")
print("Each run takes ~5-10 minutes with 10K series.\n")

input("Press Enter to start...")

results = []

for name, flags in configs:
    print("\n" + "="*70)
    print(f"Running: {name}")
    print(f"Command: python scripts/train_local.py {' '.join(flags)}")
    print("="*70 + "\n")
    
    # Run training
    cmd = ["python", "scripts/train_local.py"] + flags
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract AUC from output
    output = result.stdout
    
    # Look for CV AUC line
    auc_match = re.search(r'CV AUC: ([\d.]+) ± ([\d.]+)', output)
    if auc_match:
        mean_auc = float(auc_match.group(1))
        std_auc = float(auc_match.group(2))
        results.append((name, mean_auc, std_auc, len(flags)))
        print(f"\n✓ {name}: {mean_auc:.4f} ± {std_auc:.4f}")
    else:
        print(f"\n✗ {name}: Could not extract AUC")
        results.append((name, None, None, len(flags)))

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n{'Configuration':<25} {'AUC':<12} {'Improvement':<15}")
print("-" * 70)

baseline_auc = None
for name, mean_auc, std_auc, _ in results:
    if mean_auc is not None:
        auc_str = f"{mean_auc:.4f} ± {std_auc:.4f}"
        
        if baseline_auc is None:
            baseline_auc = mean_auc
            improvement = "-"
        else:
            improvement = f"+{(mean_auc - baseline_auc):.4f}"
        
        print(f"{name:<25} {auc_str:<12} {improvement:<15}")
    else:
        print(f"{name:<25} {'FAILED':<12} {'-':<15}")

print("\n" + "="*70)

if baseline_auc is not None:
    final_auc = results[-1][1]
    if final_auc is not None:
        total_improvement = final_auc - baseline_auc
        print(f"\nTotal improvement: +{total_improvement:.4f} AUC")
        print(f"From {baseline_auc:.4f} → {final_auc:.4f}")
        
        if total_improvement >= 0.05:
            print("\n✅ EXCELLENT! Phase 1 features are working!")
        elif total_improvement >= 0.03:
            print("\n✓ GOOD! Modest improvement from Phase 1.")
        elif total_improvement >= 0.01:
            print("\n⚠️  MARGINAL improvement. May need more features.")
        else:
            print("\n❌ NO improvement. Data may be different from last year.")

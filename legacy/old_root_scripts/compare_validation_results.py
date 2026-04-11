"""
Compare validation results before and after optimization.

Usage:
  python compare_validation_results.py before.json after.json
"""
import json
import sys

def compare(before_file, after_file, tolerance=1e-4):
    with open(before_file) as f:
        before = json.load(f)
    with open(after_file) as f:
        after = json.load(f)
    
    all_match = True
    
    for key in before:
        if key not in after:
            print(f"[MISSING] Key '{key}' not in after results")
            all_match = False
            continue
        
        if isinstance(before[key], list):
            for i, (b, a) in enumerate(zip(before[key], after[key])):
                if isinstance(b, dict):
                    for k in b:
                        diff = abs(b[k] - a[k])
                        if diff > tolerance:
                            print(f"[DIFF] {key}[{i}].{k}: {b[k]} -> {a[k]} (diff={diff})")
                            all_match = False
                else:
                    diff = abs(b - a)
                    if diff > tolerance:
                        print(f"[DIFF] {key}[{i}]: {b} -> {a} (diff={diff})")
                        all_match = False
        elif isinstance(before[key], dict):
            for k in before[key]:
                diff = abs(before[key][k] - after[key][k])
                if diff > tolerance:
                    print(f"[DIFF] {key}.{k}: {before[key][k]} -> {after[key][k]} (diff={diff})")
                    all_match = False
    
    if all_match:
        print("[PASS] All values match within tolerance")
    return all_match

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_validation_results.py before.json after.json")
        sys.exit(1)
    
    success = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)



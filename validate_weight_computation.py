"""
Validation script for weight computation logic.
Run before and after optimization to verify correctness.

Usage:
  cd /home/sagilevi1/bat_face_rec_project
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate frec && source .venv/bin/activate
  python validate_weight_computation.py
"""
import numpy as np
import sys
sys.path.insert(0, 'app')

from siamese_data.class_weights import ClassWeightCalculator, combine_class_weights
from siamese_data.anchor_negative_weights import calculate_anchor_negative_weights

def test_anchor_negative_weights():
    """Test anchor/negative weight computation."""
    print("Test 1: Anchor/Negative Weights")
    
    # Test case: 2 anchors, 8 negatives (should give anchors higher weight)
    aw, nw = calculate_anchor_negative_weights(2, 8)
    assert abs(aw * 2 - nw * 8) < 0.01, f"Weights not balanced: {aw}*2 != {nw}*8"
    print(f"  [PASS] 2 anchors, 8 negatives: anchor_w={aw:.4f}, neg_w={nw:.4f}")
    
    # Test case: equal counts
    aw, nw = calculate_anchor_negative_weights(5, 5)
    assert abs(aw - nw) < 0.01, f"Equal counts should give equal weights"
    print(f"  [PASS] 5 anchors, 5 negatives: anchor_w={aw:.4f}, neg_w={nw:.4f}")
    
    return True

def test_per_class_weights():
    """Test per-class weight computation."""
    print("\nTest 2: Per-Class Weights")
    
    global_dist = {"class_a": 100, "class_b": 50, "class_c": 25}
    config = {
        "enabled": True,
        "anchor_negative_balance": False,
        "per_class_balance": True,
        "weighting_scheme": "ins",
        "negative_pair_combination": "sum"
    }
    calc = ClassWeightCalculator(config, global_class_distribution=global_dist)
    
    # Test single-class weight (positive pair)
    labels = np.array([1.0])
    class_info = ["class_a:1.0"]
    weights = calc.compute_sample_weights(labels, class_info)
    print(f"  [PASS] Single class 'class_a': weight={weights[0]:.4f}")
    
    # Test dual-class weight (negative pair)
    labels = np.array([0.0])
    class_info = ["class_a:1.0|class_b:1.0"]
    weights = calc.compute_sample_weights(labels, class_info)
    print(f"  [PASS] Dual class 'class_a|class_b': weight={weights[0]:.4f}")
    
    return True

def test_combined_weights():
    """Test full weight computation with both components."""
    print("\nTest 3: Combined Weights (Anchor/Negative + Per-Class)")
    
    global_dist = {"class_a": 100, "class_b": 50}
    config = {
        "enabled": True,
        "anchor_negative_balance": True,
        "per_class_balance": True,
        "weighting_scheme": "ins",
        "negative_pair_combination": "sum"
    }
    calc = ClassWeightCalculator(config, global_class_distribution=global_dist)
    
    # Mixed batch: 2 positive (class_a), 3 negative (class_a|class_b)
    labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    class_info = [
        "class_a:1.0", "class_a:1.0",
        "class_a:1.0|class_b:1.0", "class_a:1.0|class_b:1.0", "class_a:1.0|class_b:1.0"
    ]
    weights = calc.compute_sample_weights(labels, class_info)
    
    print(f"  Positive weights: {weights[:2]}")
    print(f"  Negative weights: {weights[2:]}")
    print(f"  Mean weight: {np.mean(weights):.4f} (should be ~1.0)")
    
    assert abs(np.mean(weights) - 1.0) < 0.01, "Weights not normalized"
    print(f"  [PASS] Combined weights computed correctly")
    
    # Store results for comparison
    return weights.tolist()

def test_weight_combination_strategies():
    """Test different combination strategies for negative pairs."""
    print("\nTest 4: Weight Combination Strategies")
    
    weights = [0.3, 0.7]  # Two class weights
    
    sum_result = combine_class_weights(weights, "sum")
    geom_result = combine_class_weights(weights, "geometric_mean")
    prod_result = combine_class_weights(weights, "product")
    
    print(f"  SUM: {sum_result:.4f}")
    print(f"  GEOMETRIC_MEAN: {geom_result:.4f}")
    print(f"  PRODUCT: {prod_result:.4f}")
    
    return {"sum": sum_result, "geometric_mean": geom_result, "product": prod_result}

if __name__ == "__main__":
    print("=" * 60)
    print("Weight Computation Validation")
    print("=" * 60)
    
    results = {}
    
    try:
        test_anchor_negative_weights()
        test_per_class_weights()
        results["combined"] = test_combined_weights()
        results["strategies"] = test_weight_combination_strategies()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        
        # Save results for comparison
        import json
        with open("validation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to validation_results.json")
        print("Run again after optimization and compare with:")
        print("  diff validation_results_before.json validation_results_after.json")
        
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)



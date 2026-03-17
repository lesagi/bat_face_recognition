# Permutation Tests Guide

## Overview

This module implements **permutation tests** for statistical significance validation of your Siamese network model's performance. Permutation tests help answer the critical question: **"Is my model's performance significantly better than random chance?"**

## What Are Permutation Tests?

Permutation tests are non-parametric statistical methods that evaluate whether observed differences or effects in data are statistically significant. Unlike traditional parametric tests (like t-tests), permutation tests make minimal assumptions about data distribution.

### How They Work

1. **Observed Performance**: You train your model normally and record metrics (F1, accuracy, precision, recall)
2. **Null Hypothesis**: Assume there's no real relationship between inputs and outputs (labels are random)
3. **Permutation Iterations**: Run multiple training iterations with randomly shuffled labels
4. **Null Distribution**: Build a distribution of performance metrics under the null hypothesis
5. **P-value Calculation**: Compare your observed metrics to the null distribution to calculate statistical significance

### Why Use Permutation Tests?

- **Validates Model Learning**: Confirms your model learned meaningful patterns, not just memorization
- **No Distribution Assumptions**: Works with any data distribution
- **Interpretable Results**: Clear p-values show statistical significance
- **Robust**: Handles non-normal data and small sample sizes well

## What Was Implemented

### Core Components

#### 1. `PermutationTest` Class (`permutation_test.py`)

The main class that orchestrates the permutation test:

- **P-value Calculation**: Uses the formula `p = (count(null >= observed) + 1) / (n + 1)` with continuity correction
- **Null Distribution Building**: Runs multiple training iterations with permuted labels
- **Results Management**: Stores and serializes test results

**Key Methods:**
- `run()`: Execute full permutation test with trainer factory
- `compute_p_value()`: Calculate p-value from observed value and null distribution
- `run_from_null_distributions()`: Calculate results from pre-computed distributions

#### 2. `PermutationTrainer` Class (`permutation_trainer.py`)

A lightweight, optimized version of `SiameseNetworkTrainer` for efficient permutation iterations:

**Optimizations:**
- Disabled MLflow logging (not needed for null distribution)
- Disabled checkpoint saving (saves disk space)
- Disabled post-training automation (saliency maps, predictions)
- Reduced console output
- Returns only final metrics

**Key Features:**
- Same architecture and training logic as main trainer
- Supports all class balancing strategies
- Configurable epochs per iteration (typically 5-15 vs 80 for full training)

#### 3. `PermutationVisualizer` Class (`permutation_visualizer.py`)

Generates comprehensive visualizations and reports:

**Visualizations:**
- Null distribution histograms with observed value marked
- Combined multi-metric plots
- Summary table with all metrics
- P-value regions highlighted

**Exports:**
- JSON results file
- CSV summary
- PNG plots

#### 4. CLI Interface (`run_permutation_test.py`)

Command-line tool for running permutation tests:

**Features:**
- Load observed metrics from JSON files
- Load observed metrics from MLflow runs
- Manual metric specification
- Configurable test parameters
- Automatic report generation

### Configuration

Added to `app/config/config.yml`:

```yaml
statistical_tests:
  permutation_test:
    n_permutations: 100          # Number of permutation iterations
    permutation_epochs: 10        # Epochs per iteration
    significance_level: 0.05      # P-value threshold
    metrics_to_test:              # Metrics to evaluate
      - "f1"
      - "accuracy"
      - "precision"
      - "recall"
    save_null_distribution: true   # Save full distributions
    
    # Optimized mode settings (use --mode optimized)
    # Designed for memory-constrained runs while preserving statistical validity
    optimized_mode:
      n_permutations: 100         # Minimum for p<0.01 capability
      permutation_epochs: 10       # Enough for convergence
      sample_fraction: 1.0         # Full data (sampling invalidates test)
      shuffle_buffer_fraction: 0.5 # Smaller shuffle buffer
      optimizer: "sgd"             # SGD uses ~3GB less GPU memory than Adam
      gradient_accumulation_steps: 4  # Reduces peak memory
```

## How to Use

### Prerequisites

1. Train your Siamese network model normally
2. Record the final test metrics (F1, accuracy, precision, recall)
3. Ensure you have the training data available

### Method 1: Using the CLI (Recommended)

#### Basic Usage

```bash
# Activate your environment
source .venv/bin/activate  # or conda activate <env>

# Run with manually specified metrics
python -m app.statistical_tests.run_permutation_test \
  --observed-f1 0.85 \
  --observed-accuracy 0.82 \
  --observed-precision 0.88 \
  --observed-recall 0.82 \
  --n-permutations 100 \
  --output-dir evaluations/permutation_tests/my_test
```

#### Load from JSON File

Create a JSON file with your observed metrics:

```json
{
  "f1": 0.85,
  "accuracy": 0.82,
  "precision": 0.88,
  "recall": 0.82
}
```

Then run:

```bash
python -m app.statistical_tests.run_permutation_test \
  --observed-metrics my_metrics.json \
  --n-permutations 100
```

#### Load from MLflow Run

```bash
python -m app.statistical_tests.run_permutation_test \
  --mlflow-run-id abc123def456 \
  --n-permutations 100
```

#### Full CLI Options

```bash
python -m app.statistical_tests.run_permutation_test --help
```

**Key Options:**
- `--mode`: Test mode - `normal` (full data) or `optimized` (faster, sampled data)
- `--n-permutations`: Number of iterations (default: 100, recommended: 100-1000)
- `--permutation-epochs`: Epochs per iteration (default: 10, recommended: 5-15)
- `--significance-level`: P-value threshold (default: 0.05)
- `--bat-type`: 'r' for rousettus, 'm' for mauritius
- `--output-dir`: Where to save results
- `--no-plots`: Skip visualization generation
- `--verbose`: Enable detailed output

#### Optimized Mode (Memory-Efficient)

For memory-constrained environments (e.g., GPU OOM issues), use `--mode optimized`:

```bash
# Memory-efficient run with full statistical validity
python -m app.statistical_tests.run_permutation_test \
  --mlflow-run-id abc123def456 \
  --mode optimized \
  --verbose
```

**Optimized mode features (configurable in `config.yml`):**
- SGD optimizer instead of Adam (saves ~3GB GPU memory)
- Gradient accumulation (4 steps) to reduce peak memory
- Smaller shuffle buffer (50%)
- Full data and permutation count preserved for statistical validity

| Mode | Optimizer | Grad Accum | Memory Savings | Statistical Validity |
|------|-----------|------------|----------------|---------------------|
| Normal | Adam | 1 | Baseline | Full |
| Optimized | SGD | 4 | ~3-5GB GPU | Full |

**When to use each mode:**
- **Normal**: When you have sufficient GPU memory (~24GB+)
- **Optimized**: When experiencing OOM errors, or on smaller GPUs

### Statistical Validity Requirements

The permutation test p-value resolution depends on the number of permutations:

| Desired p-value | Minimum n_permutations |
|-----------------|------------------------|
| p < 0.05        | 20                     |
| p < 0.01        | 100                    |
| p < 0.001       | 1000                   |

**Important:** The default `n_permutations=100` allows detecting significance at p < 0.01.

**Warning about data sampling:** If you use `sample_fraction < 1.0`, the observed metrics 
must also come from the same sampled data subset. Otherwise, you're comparing metrics from 
different data distributions, which invalidates the statistical test.

### Method 2: Using Python API

```python
from app.statistical_tests import (
    PermutationTest,
    create_permutation_trainer,
    PermutationVisualizer
)

# Your observed metrics from training
observed_metrics = {
    "f1": 0.85,
    "accuracy": 0.82,
    "precision": 0.88,
    "recall": 0.82
}

# Create permutation test
perm_test = PermutationTest(
    n_permutations=100,
    permutation_epochs=10,
    significance_level=0.05,
    verbose=True
)

# Run the test
results = perm_test.run(
    observed_metrics=observed_metrics,
    trainer_factory=create_permutation_trainer,
    bat_type='r',
    augmented_data=False,
    data_source='video'
)

# Generate visualizations
visualizer = PermutationVisualizer(
    results,
    output_dir='evaluations/permutation_tests'
)
visualizer.generate_full_report(show=False)

# Print summary
results.print_summary()
```

### Method 3: Using Pre-computed Null Distributions

If you've already run permutations and want to test different observed values:

```python
from app.statistical_tests import PermutationTest

# Load pre-computed null distributions
null_distributions = {
    "f1": [0.52, 0.48, 0.51, ...],  # List of 100 values
    "accuracy": [0.49, 0.51, 0.50, ...]
}

# New observed metrics
observed_metrics = {"f1": 0.85, "accuracy": 0.82}

# Calculate results
perm_test = PermutationTest(significance_level=0.05)
results = perm_test.run_from_null_distributions(
    observed_metrics,
    null_distributions
)

results.print_summary()
```

## Understanding Results

### Output Files

After running a permutation test, you'll get:

```
evaluations/permutation_tests/
├── permutation_results.json      # Full results (JSON)
├── permutation_results.csv        # Summary table (CSV)
├── permutation_summary.png        # Visual summary table
├── null_distributions_all.png     # All metrics combined
├── null_dist_f1.png               # F1 distribution
├── null_dist_accuracy.png          # Accuracy distribution
├── null_dist_precision.png         # Precision distribution
└── null_dist_recall.png            # Recall distribution
```

### Reading the Results

#### Summary Table

```
Permutation Test Results (n=100)
======================================================================
Metric      Observed    Mean(Null)    Std(Null)    P-value    Significant
--------    --------    ----------    ---------    -------    -----------
f1          0.847       0.512         0.041        0.001      Yes (p<0.05)
accuracy    0.831       0.498         0.038        0.001      Yes (p<0.05)
precision   0.862       0.523         0.045        0.002      Yes (p<0.05)
recall      0.833       0.501         0.042        0.001      Yes (p<0.05)
```

**Key Columns:**
- **Observed**: Your model's actual performance
- **Mean(Null)**: Average performance with random labels (~0.5 for binary classification)
- **Std(Null)**: Standard deviation of null distribution
- **P-value**: Probability of observing this performance by chance
- **Significant**: Whether p-value < significance level (default 0.05)

#### Interpreting P-values

- **p < 0.05**: Statistically significant - model learned meaningful patterns
- **p >= 0.05**: Not significant - performance could be due to chance

**Example Interpretations:**

| P-value | Interpretation |
|---------|----------------|
| p = 0.001 | Very strong evidence against null hypothesis. Model definitely learned. |
| p = 0.01 | Strong evidence. Model likely learned meaningful patterns. |
| p = 0.05 | Borderline significant. Model probably learned, but be cautious. |
| p = 0.10 | Not significant. Model performance could be random. |
| p = 0.50 | No evidence. Model performance is consistent with random chance. |

#### Distribution Plots

The histogram plots show:
- **Blue bars**: Null distribution (performance with random labels)
- **Red dashed line**: Your observed performance
- **Gray dotted line**: Mean of null distribution
- **Red shaded area**: P-value region (more extreme than observed)

**What to Look For:**
- If observed value is far to the right of null distribution → Strong significance
- If observed value overlaps with null distribution → Not significant
- If null distribution is centered around 0.5 → Expected for binary classification

## Best Practices

### Choosing Parameters

#### Number of Permutations (`n_permutations`)

- **Quick check**: 50-100 iterations (~30-60 minutes)
- **Final validation**: 500-1000 iterations (~3-6 hours)
- **Publication quality**: 1000+ iterations

More iterations = more accurate p-values, but longer runtime.

#### Epochs per Permutation (`permutation_epochs`)

- **Minimum**: 5 epochs (very fast, approximate)
- **Recommended**: 10-15 epochs (good balance)
- **Maximum**: 20 epochs (more accurate but slower)

The goal is to see if a model *can* learn anything, not to fully train it.

### When to Run Permutation Tests

1. **After Initial Training**: Validate that your model learned something meaningful
2. **Before Publication**: Demonstrate statistical significance
3. **Model Comparison**: Compare significance between different architectures
4. **Hyperparameter Tuning**: Ensure improvements are statistically significant
5. **Data Quality Check**: If p-value is high, might indicate data issues

### Common Issues and Solutions

#### Issue: High P-values (Not Significant)

**Possible Causes:**
- Model didn't actually learn (check training curves)
- Too few epochs in permutation iterations
- Data leakage or label issues
- Model architecture too simple

**Solutions:**
- Increase `permutation_epochs` to 15-20
- Check training data quality
- Verify model architecture is appropriate
- Review training logs for issues

#### Issue: Very Low P-values (< 0.001)

**This is Good!** Your model is highly significant. Consider:
- Running more permutations for publication-quality results
- Documenting this in your paper/report

#### Issue: Long Runtime

**Optimizations:**
- Reduce `n_permutations` for quick checks
- Reduce `permutation_epochs` (minimum 5)
- Use smaller batch sizes in config
- Run on GPU (automatically used if available)

## Example Workflow

### Complete Example: Validating a Trained Model

```bash
# Step 1: Train your model normally
python -m app.siamese_training.train_siamese \
  --bat-type r \
  --data-source video \
  --epochs 80

# Step 2: Note the final test metrics from training output
# Example: F1=0.85, Accuracy=0.82, Precision=0.88, Recall=0.82

# Step 3: Run permutation test
python -m app.statistical_tests.run_permutation_test \
  --observed-f1 0.85 \
  --observed-accuracy 0.82 \
  --observed-precision 0.88 \
  --observed-recall 0.82 \
  --n-permutations 100 \
  --permutation-epochs 10 \
  --output-dir evaluations/permutation_tests/validation_run_1

# Step 4: Review results
# Check permutation_summary.png and null_distributions_all.png
# Read permutation_results.json for detailed statistics

# Step 5: If significant (p < 0.05), your model learned!
# If not significant, investigate training issues
```

### Python Script Example

```python
#!/usr/bin/env python3
"""Example: Run permutation test after training."""

from app.statistical_tests import (
    PermutationTest,
    create_permutation_trainer,
    PermutationVisualizer
)

def validate_model_performance():
    # Metrics from your trained model
    observed = {
        "f1": 0.847,
        "accuracy": 0.831,
        "precision": 0.862,
        "recall": 0.833
    }
    
    # Run permutation test
    perm_test = PermutationTest(
        n_permutations=100,
        permutation_epochs=10,
        significance_level=0.05,
        verbose=True
    )
    
    results = perm_test.run(
        observed_metrics=observed,
        trainer_factory=create_permutation_trainer,
        bat_type='r',
        data_source='video'
    )
    
    # Generate report
    visualizer = PermutationVisualizer(
        results,
        output_dir='evaluations/permutation_tests/example'
    )
    visualizer.generate_full_report()
    
    # Check significance
    significant_metrics = [
        m for m, r in results.metrics.items() 
        if r.significant
    ]
    
    if significant_metrics:
        print(f"✅ Model is significant for: {significant_metrics}")
        return True
    else:
        print("❌ Model is NOT statistically significant")
        return False

if __name__ == '__main__':
    validate_model_performance()
```

## Integration with Existing Workflow

### MLflow Integration

The permutation test results can be logged to MLflow:

```python
import mlflow
from app.statistical_tests import PermutationTest

# Run permutation test
results = perm_test.run(...)

# Log to MLflow
with mlflow.start_run():
    mlflow.log_params({
        "permutation_n": results.n_permutations,
        "permutation_epochs": results.permutation_epochs
    })
    
    for metric, result in results.metrics.items():
        mlflow.log_metric(f"permutation_{metric}_pvalue", result.p_value)
        mlflow.log_metric(f"permutation_{metric}_significant", 
                         int(result.significant))
    
    # Log results file as artifact
    results.save("permutation_results.json")
    mlflow.log_artifact("permutation_results.json")
```

### Automated Validation

Add to your training pipeline:

```python
# After training completes
if config.get("run_permutation_test", False):
    from app.statistical_tests import PermutationTest, create_permutation_trainer
    
    perm_test = PermutationTest(n_permutations=100)
    results = perm_test.run(
        observed_metrics={
            "f1": final_f1,
            "accuracy": final_accuracy,
            "precision": final_precision,
            "recall": final_recall
        },
        trainer_factory=create_permutation_trainer,
        bat_type=bat_type,
        data_source=data_source
    )
    
    # Fail if not significant
    if not any(r.significant for r in results.metrics.values()):
        raise ValueError("Model performance is not statistically significant!")
```

## Troubleshooting

### Import Errors

```python
# Make sure you're in the project directory
import sys
sys.path.insert(0, 'app')

from statistical_tests import PermutationTest
```

### TensorFlow/GPU Issues

The permutation trainer uses TensorFlow. Ensure:
- TensorFlow is installed: `pip install tensorflow`
- GPU is available (optional but recommended)
- CUDA/cuDNN configured if using GPU

### Memory Issues

If running out of memory:
- Reduce batch size in `config.yml`
- Reduce `n_permutations` (run fewer iterations)
- Use CPU instead of GPU (slower but less memory)

### Long Runtime

Permutation tests are computationally expensive:
- 100 iterations × 10 epochs ≈ 1-3 hours (depending on data size)
- Use GPU if available
- Consider running overnight for large tests
- Use `--permutation-epochs 5` for quick checks

## References

- [Permutation Tests in Machine Learning - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/permutation-tests-in-machine-learning/)
- Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research*, 11(6).

## Support

For issues or questions:
1. Check this guide first
2. Review the code comments in the module files
3. Check example outputs in `evaluations/permutation_tests/`

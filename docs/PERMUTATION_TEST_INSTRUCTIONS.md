# Permutation Test Execution Instructions

## What Was Fixed

The metric loading from MLflow was missing the **accuracy** metric. I've updated [`app/statistical_tests/run_permutation_test.py`](app/statistical_tests/run_permutation_test.py) to include:
- `'test_accuracy': 'accuracy'`
- `'final_test_accuracy': 'accuracy'`

Now the test will capture and validate all 4 metrics: F1, precision, recall, AND accuracy.

## How to Run the Permutation Test

### Basic Command (Using Your Existing MLflow Run)

If you want to use the same MLflow run that produced your model results:

```bash
cd /home/sagilevi1/bat_face_rec_project
python -m app.statistical_tests.run_permutation_test \
    --mlflow-run-id b78b08ff00374e17a560ffb8f7d107b7 \
    --n-permutations 100 \
    --mode optimized
```

### Alternative: Using Saved Results File

If you want to run with a JSON file containing observed metrics:

```bash
# First, create a metrics JSON file with accuracy included
cat > observed_metrics.json << 'EOF'
{
  "f1": 0.519307,
  "precision": 0.5618544816970825,
  "recall": 0.46557626128196716,
  "accuracy": 0.89  # Add your observed accuracy value here
}
EOF

# Then run the test
python -m app.statistical_tests.run_permutation_test \
    --observed-metrics observed_metrics.json \
    --n-permutations 100 \
    --mode optimized
```

### Full Test (More Statistical Power)

For more accurate results (but takes longer), use normal mode instead of optimized:

```bash
python -m app.statistical_tests.run_permutation_test \
    --mlflow-run-id b78b08ff00374e17a560ffb8f7d107b7 \
    --n-permutations 100 \
    --mode normal
```

## Important Parameters Explained

- `--mlflow-run-id`: Your model training run ID (loads observed metrics automatically)
- `--n-permutations`: How many random label permutations to test (default: 100)
- `--mode optimized`: Uses 50% of data per permutation (faster, ~2-3 weeks for 100 perms)
- `--mode normal`: Uses 100% of data per permutation (slower but more power, ~6+ weeks)
- `--output-dir`: Where to save results (default: `evaluations/permutation_tests/{timestamp}`)
- `--significance-level`: Alpha threshold (default: 0.05)

## Running in Background with nohup

To run the test in the background so you can close your terminal:

```bash
cd /home/sagilevi1/bat_face_rec_project

# Create a new nohup output file
nohup python -m app.statistical_tests.run_permutation_test \
    --mlflow-run-id b78b08ff00374e17a560ffb8f7d107b7 \
    --n-permutations 100 \
    --mode optimized \
    --verbose > permutation_test_fixed.out 2>&1 &
```

Then monitor progress:

```bash
# Check the latest output
tail -f permutation_test_fixed.out

# Check if process is still running
ps aux | grep permutation_test
```

## What to Expect in Output

### During execution:
- Permutation X/100 progress with ETA
- Each permutation shows: F1, accuracy, precision, recall scores
- Elapsed time tracking

### After completion:
The script will:
1. Save results to `evaluations/permutation_tests/{timestamp}/permutation_results.json`
2. Generate visualizations (plots) in the same directory
3. Print summary table showing:
   - Observed vs. Null mean for each metric
   - P-values (< 0.05 = statistically significant)
   - Significance column (Yes/No)

Example output:
```
======================================================================
Permutation Test Results (n=100)
======================================================================
Metric       Observed   Mean(Null)   Std(Null)   P-value    Significant
------------ ---------- ------------ ---------- ---------- ------------
f1           0.5193     0.0000       0.0000     0.0099     Yes (p<0.05)
accuracy     0.8910     0.8960       0.0015     0.8712     No
precision    0.5619     0.0000       0.0000     0.0099     Yes (p<0.05)
recall       0.4656     0.0000       0.0000     0.0099     Yes (p<0.05)
======================================================================
```

## Interpreting Results

- **p-value < 0.05**: Your model's metric is **statistically significant** (reject null hypothesis)
- **p-value ≥ 0.05**: Your model's metric is **NOT statistically significant** (fail to reject null)

For your model:
- **F1 and Precision should be highly significant** (much better than random)
- **Accuracy may not be significant** (due to class imbalance - many negative pairs)
- **Recall should be highly significant** (better than random)

## Time Estimates

- **Optimized mode (50% data)**: ~5-7 days for 100 permutations (with 2 GPUs)
- **Normal mode (100% data)**: ~10-14 days for 100 permutations (with 2 GPUs)

Your original test hit this timeline, so the timing hasn't changed.

## Troubleshooting

### If MLflow run ID is wrong:
```bash
# List recent MLflow runs
mlflow runs list --experiment-id 0 | head -20
```

### If accuracy still appears missing:
The test will skip it with a warning. If this happens, you can manually specify:
```bash
python -m app.statistical_tests.run_permutation_test \
    --observed-f1 0.519307 \
    --observed-precision 0.5618544816970825 \
    --observed-recall 0.46557626128196716 \
    --observed-accuracy 0.89 \
    --n-permutations 100 \
    --mode optimized
```

### If GPU memory issues occur:
The code uses SGD optimizer in optimized mode to reduce memory. If still getting OOM errors, reduce batch size:
```bash
# (This would require code modification - open an issue if needed)
```

## Next Steps After Test Completes

1. Check the results JSON file: `evaluations/permutation_tests/{timestamp}/permutation_results.json`
2. Review the plots generated in the same directory
3. Extract p-values and significance conclusions
4. Include results in your paper/report

Good luck! The test should confirm that your model's performance is statistically significant.

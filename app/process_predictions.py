#!/usr/bin/env python3
"""
Process prediction results to create a normalized confusion matrix.

This script loads prediction CSV files, creates a pivot table of y_hat values,
normalizes each column to sum to 100%, and adds Grand Total rows/columns.

Usage:
    python process_predictions.py --input predictions.csv --output processed_matrix.csv
    python process_predictions.py -i predictions.csv -o processed_matrix.csv
    python process_predictions.py --input predictions.csv  # Auto-generates output filename
"""

import argparse
import os
import pandas as pd
import sys


def process_predictions(input_path, output_path=None):
    """
    Process prediction results and create normalized confusion matrix.

    Args:
        input_path (str): Path to input CSV file with predictions
        output_path (str, optional): Path for output CSV file. If None, auto-generates.

    Returns:
        str: Path to the output file
    """
    # Validate input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"📄 Loading predictions from: {input_path}")

    # Load the CSV file
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Successfully loaded {len(df)} records")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Validate required columns
    required_columns = ["y_hat", "input1_name", "input2_name"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"📊 Data shape: {df.shape}")
    print(f"🏷️  Unique input1_name values: {df['input1_name'].nunique()}")
    print(f"🏷️  Unique input2_name values: {df['input2_name'].nunique()}")

    # Step 1: Create the pivot table (raw sum of y_hat)
    print("🔄 Creating pivot table...")
    pivot = pd.pivot_table(
        df,
        values="y_hat",
        index="input1_name",
        columns="input2_name",
        aggfunc="sum",
        fill_value=0,
    )

    print(f"📋 Pivot table shape: {pivot.shape}")

    # Step 2: Normalize each column to sum to 100%
    print("🔢 Normalizing columns to percentages...")
    normalized = pivot.div(pivot.sum(axis=0), axis=1) * 100

    # Step 3: Add "Grand Total" row and column
    print("➕ Adding Grand Total row and column...")
    normalized.loc["Grand Total"] = normalized.sum(axis=0)
    normalized["Grand Total"] = normalized.sum(axis=1)

    # Round values to two decimals for readability
    final_result = normalized.round(2)

    # Generate output path if not provided
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(
            input_dir, f"processed_confusion_matrix_{input_name}.csv"
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save to CSV file
    print(f"💾 Saving processed matrix to: {output_path}")
    final_result.to_csv(output_path)

    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print(f"   • Matrix dimensions: {final_result.shape}")
    print(f"   • Non-zero values: {(final_result != 0).sum().sum()}")
    print(f"   • Maximum value: {final_result.max().max():.2f}%")
    print(
        f"   • Minimum non-zero value: {final_result[final_result > 0].min().min():.2f}%"
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Process prediction results to create normalized confusion matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with auto-generated output filename
    python process_predictions.py --input predictions_20240815_004639.csv
    
    # Specify custom output filename
    python process_predictions.py --input predictions_20240815_004639.csv --output my_matrix.csv
    
    # Using short argument names
    python process_predictions.py -i predictions.csv -o matrix.csv
    
    # Process file with full path
    python process_predictions.py --input /path/to/predictions.csv --output /path/to/output.csv
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input CSV file containing prediction results",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Path for output CSV file (optional, auto-generates if not provided)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        # Process the predictions
        output_file = process_predictions(args.input, args.output)

        print(f"\n✅ Successfully processed predictions!")
        print(f"📁 Output saved to: {output_file}")

        if args.verbose:
            print(f"\n📄 Preview of processed matrix:")
            result_df = pd.read_csv(output_file, index_col=0)
            print(result_df.head(10))

    except Exception as e:
        print(f"❌ Error processing predictions: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

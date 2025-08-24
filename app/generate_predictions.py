#!/usr/bin/env python3
"""
Generate prediction results for Siamese network model evaluation.

This script loads a trained Siamese model, processes image pairs from input directories,
and generates a CSV file with prediction results including confidence scores and metadata.

Usage:
    python generate_predictions.py --model /path/to/model --input /path/to/data
    python generate_predictions.py -m model_path -i data_path --verbose
"""

import os
import sys

# Fix TensorFlow mutex lock issues on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import csv
import numpy as np
from datetime import datetime

# Add visualization libraries
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add the app directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.image_utils import is_img_file


def preprocess_siamese_input_flexible(file_path, target_size=224):
    """
    Preprocess a single image file path into a tensor with flexible target size.

    Args:
        file_path (str): Path to the image file
        target_size (int): Target size for resizing (assumes square images)

    Returns:
        tf.Tensor: Preprocessed image tensor
    """
    # Import TensorFlow only when needed
    import tensorflow as tf
    
    try:
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image - try both PNG and JPEG
        try:
            img = tf.io.decode_png(byte_img)
        except:
            img = tf.io.decode_jpeg(byte_img)

        # Preprocessing steps - resizing the image to target size
        img = tf.image.resize(img, (target_size, target_size))

        # Scale image to be between 0 and 1
        img = img / 255.0

        # Ensure the image has 3 channels
        img = tf.image.convert_image_dtype(img, tf.float32)
        if tf.shape(img)[2] == 1:  # Grayscale
            img = tf.repeat(img, 3, axis=2)
        elif tf.shape(img)[2] == 4:  # RGBA
            img = img[:, :, :3]

        return img
    except Exception as e:
        tf.print(f"Error processing {file_path}: {e}")
        # Return a black image as fallback
        return tf.zeros((target_size, target_size, 3), dtype=tf.float32)


def load_siamese_model(model_path):
    """
    Load a trained Siamese model from the specified path.

    Args:
        model_path (str): Path to the trained model directory

    Returns:
        tf.keras.Model: Loaded Siamese model
    """
    # Import TensorFlow and L1Dist only when needed
    import tensorflow as tf
    from siamese_network.network import L1Dist
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "L1Dist": L1Dist,
                "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
            },
        )
        print(f"✅ Successfully loaded model from: {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {e}")


def create_siamese_confusion_matrix_plot(
    binary_true,
    binary_pred,
    class_verification_data,
    output_path,
    title="Siamese Network Evaluation",
    show_percentages=True,
):
    """
    Create confusion matrix plots appropriate for Siamese networks.

    Args:
        binary_true (list): Binary true labels ("Same", "Different")
        binary_pred (list): Binary predicted labels ("Same", "Different")
        class_verification_data (dict): Per-class verification statistics
        output_path (str): Path to save the plot
        title (str): Title for the plot
        show_percentages (bool): Whether to show percentages alongside counts
    """
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Binary Confusion Matrix (Same vs Different)
    binary_labels = ["Different", "Same"]  # Order matters for confusion matrix
    cm_binary = confusion_matrix(binary_true, binary_pred, labels=binary_labels)

    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm_binary, display_labels=binary_labels
    )
    disp1.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title(
        "Binary Classification\n(Same vs Different Pairs)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Predicted", fontsize=10)
    axes[0].set_ylabel("True", fontsize=10)

    # Add percentages as text annotations if requested
    if show_percentages:
        cm_binary_norm = confusion_matrix(
            binary_true, binary_pred, labels=binary_labels, normalize="true"
        )
        for i in range(len(binary_labels)):
            for j in range(len(binary_labels)):
                percentage = cm_binary_norm[i, j] * 100
                axes[0].text(
                    j,
                    i + 0.2,
                    f"({percentage:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="darkred",
                )

    # Calculate binary metrics
    tn, fp, fn, tp = cm_binary.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Add metrics text below the confusion matrix
    metrics_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}"
    axes[0].text(
        0.5,
        -0.15,
        metrics_text,
        transform=axes[0].transAxes,
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    # Plot 2: Per-Class Verification Rates
    if class_verification_data:
        classes = list(class_verification_data.keys())
        same_class_accuracy = [
            class_verification_data[cls]["same_accuracy"] for cls in classes
        ]
        different_class_accuracy = [
            class_verification_data[cls]["different_accuracy"] for cls in classes
        ]

        x = np.arange(len(classes))
        width = 0.35

        bars1 = axes[1].bar(
            x - width / 2,
            same_class_accuracy,
            width,
            label="Same-class pairs",
            color="lightblue",
        )
        bars2 = axes[1].bar(
            x + width / 2,
            different_class_accuracy,
            width,
            label="Different-class pairs",
            color="lightcoral",
        )

        axes[1].set_xlabel("Bat Classes", fontsize=10)
        axes[1].set_ylabel("Accuracy", fontsize=10)
        axes[1].set_title(
            "Per-Class Verification Accuracy", fontsize=12, fontweight="bold"
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes, rotation=45, ha="right")
        axes[1].legend()
        axes[1].set_ylim(0, 1.0)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    axes[1].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No per-class data\navailable",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=12,
        )
        axes[1].set_title("Per-Class Verification", fontsize=12, fontweight="bold")

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Adjust layout with extra space at bottom for metrics
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Siamese confusion matrix plot saved to: {output_path}")

    # Print detailed statistics
    print(f"📈 Binary Classification Results:")
    print(f"   • Overall Accuracy: {accuracy:.3f} ({tp + tn}/{tp + tn + fp + fn})")
    print(f"   • Precision (Same): {precision:.3f}")
    print(f"   • Recall (Same): {recall:.3f}")
    print(f"   • F1-Score: {f1:.3f}")

    print(f"📊 Confusion Matrix Breakdown:")
    print(f"   • True Negatives (Different→Different): {tn}")
    print(f"   • False Positives (Different→Same): {fp}")
    print(f"   • False Negatives (Same→Different): {fn}")
    print(f"   • True Positives (Same→Same): {tp}")

    if class_verification_data:
        print(f"🔍 Per-Class Verification:")
        for cls in classes:
            same_acc = class_verification_data[cls]["same_accuracy"]
            diff_acc = class_verification_data[cls]["different_accuracy"]
            print(
                f"   • {cls}: Same-pairs {same_acc:.3f}, Different-pairs {diff_acc:.3f}"
            )

    return output_path


def generate_predictions(
    model,
    input_dir,
    output_dir=None,
    include_subdirs=None,
    verbose=False,
    max_pairs=None,
):
    """
    Generate predictions for all image pairs in the input directory.

    Expected directory structure:
        input_dir/
        ├── bat1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── bat2/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── ...

    Args:
        model: Trained Siamese model
        input_dir (str): Path to input directory containing bat directories
        output_dir (str, optional): Output directory (defaults to input_dir)
        include_subdirs (list, optional): Specific bat directories to include
        verbose (bool): Enable verbose output
        max_pairs (int, optional): Maximum number of pairs to process (for testing)

    Returns:
        tuple: (csv_path, confusion_matrix_path)
    """
    # Get model's expected input size
    input_shape = model.input_shape[0]  # First input (for Siamese networks)
    model_input_size = input_shape[1]  # Assuming square images (height = width)
    print(f"📏 Model expects input size: {model_input_size}×{model_input_size}")
    if output_dir is None:
        output_dir = input_dir

    # Validate input directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print(f"📁 Processing data from: {input_dir}")

    # Check for bat directories (single level nesting: base/bat1, base/bat2, etc.)
    bat_dirs = [
        d
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
    ]

    if not bat_dirs:
        raise ValueError(f"No bat directories found in: {input_dir}")

    # Validate that bat directories contain images
    valid_bat_dirs = []
    for bat_dir in bat_dirs:
        bat_path = os.path.join(input_dir, bat_dir)
        image_files = [
            f
            for f in os.listdir(bat_path)
            if os.path.isfile(os.path.join(bat_path, f)) and is_img_file(f)
        ]
        if image_files:
            valid_bat_dirs.append(bat_dir)
        else:
            print(f"⚠️  Warning: No images found in directory: {bat_dir}")

    if not valid_bat_dirs:
        raise ValueError(f"No valid bat directories with images found in: {input_dir}")

    # Filter to specific subdirectories if requested
    if include_subdirs:
        valid_bat_dirs = [d for d in valid_bat_dirs if d in include_subdirs]
        if not valid_bat_dirs:
            raise ValueError(
                f"None of the specified bat directories found: {include_subdirs}"
            )

    print(
        f"📂 Found {len(valid_bat_dirs)} bat directories: {valid_bat_dirs[:5]}{'...' if len(valid_bat_dirs) > 5 else ''}"
    )

    # Create data pairs directly from file paths
    print("🔄 Creating data pairs from file paths...")

    # Get all class directories (bat directories) as full paths
    class_dirs = [os.path.join(input_dir, bat_dir) for bat_dir in valid_bat_dirs]

    print(f"📂 Found {len(class_dirs)} class directories")

    # Create positive and negative pairs
    raw_data_pairs = []

    # Positive pairs (same class) - limit pairs per class to avoid explosion
    from itertools import combinations

    for class_dir in class_dirs:
        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f)) and is_img_file(f)
        ]

        # Create combinations of files from same class (limit to avoid too many pairs)
        max_positive_pairs = 10  # Limit positive pairs per class
        file_combinations = list(combinations(files, 2))

        # Take a random sample if too many combinations
        if len(file_combinations) > max_positive_pairs:
            import random

            file_combinations = random.sample(file_combinations, max_positive_pairs)

        for file1, file2 in file_combinations:
            raw_data_pairs.append((file1, file2, 1.0))

    # Negative pairs (different classes)
    import random

    for class_dir1, class_dir2 in combinations(class_dirs, 2):
        files1 = [
            os.path.join(class_dir1, f)
            for f in os.listdir(class_dir1)
            if os.path.isfile(os.path.join(class_dir1, f)) and is_img_file(f)
        ]
        files2 = [
            os.path.join(class_dir2, f)
            for f in os.listdir(class_dir2)
            if os.path.isfile(os.path.join(class_dir2, f)) and is_img_file(f)
        ]

        # Create some pairs between different classes (limit to avoid explosion)
        min_size = min(len(files1), len(files2))
        num_pairs = min(min_size, 5)  # Reduced from 20 to 5 for faster processing

        for i in range(num_pairs):
            file1 = random.choice(files1)
            file2 = random.choice(files2)
            raw_data_pairs.append((file1, file2, 0.0))

    if not raw_data_pairs:
        raise ValueError(
            "No data pairs were created. Check if the input directories contain valid images."
        )

    print(f"📊 Created {len(raw_data_pairs)} data pairs")

    # Apply max_pairs limit if specified
    if max_pairs and len(raw_data_pairs) > max_pairs:
        import random

        raw_data_pairs = random.sample(raw_data_pairs, max_pairs)
        print(f"⚡ Limited to {max_pairs} pairs for faster processing")

    # Shuffle the pairs
    random.shuffle(raw_data_pairs)

    # Import TensorFlow metrics only when needed
    import tensorflow as tf
    from tensorflow.keras.metrics import Precision, Recall
    
    # Initialize metrics
    r = Recall()
    p = Precision()

    # Initialize lists for Siamese confusion matrix (binary classification)
    binary_true_labels = []  # "Same" or "Different"
    binary_pred_labels = []  # "Same" or "Different"

    # Initialize per-class verification tracking
    class_verification = {}
    for bat_dir in valid_bat_dirs:
        class_verification[bat_dir] = {
            "same_correct": 0,
            "same_total": 0,
            "different_correct": 0,
            "different_total": 0,
        }

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.csv"
    csv_path = os.path.join(output_dir, filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"💾 Saving predictions to: {csv_path}")
    print("🚀 Starting prediction generation...")

    processed_pairs = 0
    batch_size = 32  # Increased from 16 for faster processing
    total_batches = (len(raw_data_pairs) + batch_size - 1) // batch_size

    # Generate predictions and write to CSV
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        # Write headers
        writer.writerow(
            [
                "input1_name",
                "input1_frame",
                "input2_name",
                "input2_frame",
                "y_true",
                "y_hat",
                "success",
            ]
        )

        # Process pairs in batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(raw_data_pairs))
            batch_pairs = raw_data_pairs[start_idx:end_idx]

            if verbose or batch_idx % 5 == 0:  # More frequent progress updates
                print(
                    f"  Processing batch {batch_idx + 1}/{total_batches} ({processed_pairs}/{len(raw_data_pairs)} pairs)"
                )

            # Prepare batch data
            batch_input1 = []
            batch_input2 = []
            batch_labels = []
            batch_paths = []

            for file1, file2, label in batch_pairs:
                try:
                    # Load and preprocess images with model's expected size
                    img1 = preprocess_siamese_input_flexible(
                        file1, target_size=model_input_size
                    )
                    img2 = preprocess_siamese_input_flexible(
                        file2, target_size=model_input_size
                    )

                    batch_input1.append(img1)
                    batch_input2.append(img2)
                    batch_labels.append(label)
                    batch_paths.append((file1, file2))

                except Exception as e:
                    if verbose:
                        print(f"  Warning: Error processing pair {file1}, {file2}: {e}")
                    continue

            if not batch_input1:  # Skip empty batches
                continue

            # Convert to tensors
            batch_input1 = tf.stack(batch_input1)
            batch_input2 = tf.stack(batch_input2)
            batch_labels = tf.constant(batch_labels)

            # Generate predictions
            y_hat = model.predict([batch_input1, batch_input2], verbose=0)

            # Update metrics
            r.update_state(batch_labels, y_hat)
            p.update_state(batch_labels, y_hat)

            # Write predictions to CSV
            for i in range(len(y_hat)):
                file1, file2 = batch_paths[i]

                # Extract bat names from directory structure
                bat_name1 = os.path.basename(os.path.dirname(file1))
                bat_name2 = os.path.basename(os.path.dirname(file2))

                # Extract image filenames for frame info
                input1_filename = os.path.basename(file1)
                input2_filename = os.path.basename(file2)

                # Parse filename format for frame numbers: name.frame.extension
                try:
                    name1, frame1, extension1 = input1_filename.split(".")
                    name2, frame2, extension2 = input2_filename.split(".")
                except ValueError:
                    # Handle different filename formats
                    frame1 = "0"
                    frame2 = "0"

                # Binary prediction based on threshold
                prediction = 1 if y_hat[i][0] > 0.5 else 0
                success = 1 - abs(
                    float(batch_labels[i]) - prediction
                )  # Accuracy for this pair

                # Collect data for Siamese binary confusion matrix
                true_label = "Same" if float(batch_labels[i]) == 1.0 else "Different"
                pred_label = "Same" if prediction == 1 else "Different"

                binary_true_labels.append(true_label)
                binary_pred_labels.append(pred_label)

                # Collect per-class verification statistics
                if float(batch_labels[i]) == 1.0:  # Same class pair
                    # Both images are from the same class
                    class_verification[bat_name1]["same_total"] += 1
                    if prediction == 1:  # Correctly identified as same
                        class_verification[bat_name1]["same_correct"] += 1
                else:  # Different class pair
                    # Images are from different classes
                    class_verification[bat_name1]["different_total"] += 1
                    class_verification[bat_name2]["different_total"] += 1
                    if prediction == 0:  # Correctly identified as different
                        class_verification[bat_name1]["different_correct"] += 1
                        class_verification[bat_name2]["different_correct"] += 1

                # Write row to CSV
                writer.writerow(
                    [
                        bat_name1,  # Use bat class name instead of filename
                        frame1,
                        bat_name2,  # Use bat class name instead of filename
                        frame2,
                        float(batch_labels[i]),  # Convert tensor to float
                        float(y_hat[i][0]),  # Raw confidence score
                        success,
                    ]
                )

                processed_pairs += 1

    # Calculate final metrics
    final_recall = r.result().numpy()
    final_precision = p.result().numpy()

    print(f"\n✅ Prediction generation completed!")
    print(f"📈 Results Summary:")
    print(f"   • Total pairs processed: {processed_pairs}")
    print(f"   • Recall: {final_recall:.4f}")
    print(f"   • Precision: {final_precision:.4f}")
    print(f"   • Output file: {csv_path}")

    # Create Siamese confusion matrix plot
    if binary_true_labels and binary_pred_labels:
        print("\n🎨 Creating Siamese network confusion matrix plot...")

        # Calculate per-class verification rates
        class_verification_rates = {}
        for cls, stats in class_verification.items():
            same_accuracy = (
                stats["same_correct"] / stats["same_total"]
                if stats["same_total"] > 0
                else 0
            )
            different_accuracy = (
                stats["different_correct"] / stats["different_total"]
                if stats["different_total"] > 0
                else 0
            )
            class_verification_rates[cls] = {
                "same_accuracy": same_accuracy,
                "different_accuracy": different_accuracy,
            }

        # Create plot filename
        plot_filename = f"siamese_confusion_matrix_{timestamp}.png"
        plot_path = os.path.join(output_dir, plot_filename)

        # Create the plot
        create_siamese_confusion_matrix_plot(
            binary_true_labels,
            binary_pred_labels,
            class_verification_rates,
            plot_path,
            title="Siamese Network Evaluation",
        )

        return csv_path, plot_path
    else:
        print("⚠️  No confusion matrix data collected")
        return csv_path, None


def main():
    parser = argparse.ArgumentParser(
        description="Generate prediction results for Siamese network evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - process all bat directories in data folder
    python generate_predictions.py --model /path/to/model --input /path/to/data
    
    # With custom output directory  
    python generate_predictions.py --model /path/to/model --input /path/to/data --output /path/to/output
    
    # Process specific bat directories only
    python generate_predictions.py --model /path/to/model --input /path/to/data --subdirs charlie fidu babyis
    
    # With verbose output
    python generate_predictions.py -m /path/to/model -i /path/to/data --verbose
    
    # Fast testing with limited pairs
    python generate_predictions.py -m /path/to/model -i /path/to/data --max-pairs 100
    
    # Using short argument names
    python generate_predictions.py -m model_path -i data_path -o output_path
        """,
    )

    parser.add_argument(
        "--model", "-m", required=True, help="Path to trained Siamese model directory"
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input directory containing image data",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Path to output directory for predictions (default: same as input)",
    )

    parser.add_argument(
        "--subdirs",
        nargs="*",
        help="Specific bat directories to process (e.g., bat1 bat2 charlie)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Maximum number of pairs to process (for testing/faster execution)",
    )

    args = parser.parse_args()

    try:
        # Load the model
        model = load_siamese_model(args.model)

        # Generate predictions
        result = generate_predictions(
            model=model,
            input_dir=args.input,
            output_dir=args.output,
            include_subdirs=args.subdirs,
            verbose=args.verbose,
            max_pairs=args.max_pairs,
        )

        # Handle return value (could be tuple or single path)
        if isinstance(result, tuple):
            csv_path, plot_path = result
            print(f"\n🎉 Success! Predictions saved to: {csv_path}")
            if plot_path:
                print(f"📊 Confusion matrix plot saved to: {plot_path}")
        else:
            print(f"\n🎉 Success! Predictions saved to: {result}")

    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

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
    model_version,
    bat_type,
    source,
    background,
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
        axes[1].set_ylim(0, 1.5)

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

    # Overall title with subtitle
    subtitle = f"Model v{model_version} | Type: {bat_type} | Source: {source} | Background: {background}"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=16, fontweight="bold")

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
    model_version,
    bat_type,
    source,
    background,
    include_subdirs=None,
    verbose=False,
    max_pairs=None,
):
    """
    Generate predictions for all image pairs in the input directory.

    Expected filename structure:
        input_dir/
        ├── r--W--IMG_20250518_145705--aug001.jpg
        ├── r--H--IMG_20250518_144900.jpg
        ├── r--rasmi--IMG_20250518_151950--aug003.jpg
        └── ...

    Args:
        model: Trained Siamese model
        input_dir (str): Path to input directory containing images with filename-based classes
        model_version (int): Model version number (e.g., 1, 2, 3)
        bat_type (str): Bat type ('r' for Rousettus or 'm' for Mauritius)
        source (str): Image source type ('video' or 'still')
        background (str): Background type used ('green', 'random', or 'original')
        include_subdirs (list, optional): Specific bat classes to include (e.g., ['W', 'H', 'rasmi'])
        verbose (bool): Enable verbose output
        max_pairs (int, optional): Maximum number of pairs to process (for testing)

    Returns:
        tuple: (csv_path, confusion_matrix_path)
    """
    # Get model's expected input size
    input_shape = model.input_shape[0]  # First input (for Siamese networks)
    model_input_size = input_shape[1]  # Assuming square images (height = width)
    print(f"📏 Model expects input size: {model_input_size}×{model_input_size}")
    
    # Set fixed output directory relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from app/
    output_dir = os.path.join(project_root, "evaluations")

    # Validate input directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print(f"📁 Processing data from: {input_dir}")

    # Import the filename parsing function and splitter
    from siamese_network.data_splitter import parse_filename_class, SiameseNetworkTrainingDataSplitter

    # Use the training splitter to create pairs with filenames; use all data in training
    print("🔄 Creating data pairs using SiameseNetworkTrainingDataSplitter...")
    splitter = SiameseNetworkTrainingDataSplitter(
        images_dirs_paths_list=[input_dir],
        training_portion=1.0,
        mode="combination",
        skip_preprocessing=True,
    )

    # Convert dataset elements to python types
    raw_data_pairs = []
    for f1_b, f2_b, lbl, class_info in splitter.train_data.as_numpy_iterator():
        f1 = f1_b.decode("utf-8") if isinstance(f1_b, (bytes, bytearray)) else str(f1_b)
        f2 = f2_b.decode("utf-8") if isinstance(f2_b, (bytes, bytearray)) else str(f2_b)
        raw_data_pairs.append((f1, f2, float(lbl)))

    # Optional class filtering at pair level
    if include_subdirs:
        def _cls_from_path(p):
            fname = os.path.basename(p)
            parsed = parse_filename_class(fname)
            return parsed[1] if parsed else None
        filtered = []
        for f1, f2, lbl in raw_data_pairs:
            c1 = _cls_from_path(f1)
            c2 = _cls_from_path(f2)
            if c1 in include_subdirs and c2 in include_subdirs:
                filtered.append((f1, f2, lbl))
        raw_data_pairs = filtered

    if not raw_data_pairs:
        raise ValueError("No data pairs were created by the splitter (after optional filtering).")

    print(f"📊 Created {len(raw_data_pairs)} data pairs")

    # Apply max_pairs limit if specified
    import random
    if max_pairs and len(raw_data_pairs) > max_pairs:
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

    # Initialize per-class verification tracking from observed classes in pairs
    observed_classes = set()
    for f1, f2, _ in raw_data_pairs:
        fn1 = os.path.basename(f1)
        fn2 = os.path.basename(f2)
        p1 = parse_filename_class(fn1)
        p2 = parse_filename_class(fn2)
        if p1:
            observed_classes.add(p1[1])
        if p2:
            observed_classes.add(p2[1])
    class_verification = {
        cls: {
            "same_correct": 0,
            "same_total": 0,
            "different_correct": 0,
            "different_total": 0,
        }
        for cls in observed_classes
    }

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_v{model_version}_{bat_type}_{source}_{background}_{timestamp}.csv"
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

                # Extract bat names from filename parsing
                filename1 = os.path.basename(file1)
                filename2 = os.path.basename(file2)
                
                parsed1 = parse_filename_class(filename1)
                parsed2 = parse_filename_class(filename2)
                
                if parsed1 and parsed2:
                    _, bat_name1, file_id1, aug_suffix1 = parsed1
                    _, bat_name2, file_id2, aug_suffix2 = parsed2
                    
                    # Use file_id as frame info
                    frame1 = file_id1
                    frame2 = file_id2
                else:
                    # Fallback to filename parsing
                    bat_name1 = os.path.splitext(filename1)[0]
                    bat_name2 = os.path.splitext(filename2)[0]
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
                        bat_name1,  # Use bat class name from filename
                        frame1,
                        bat_name2,  # Use bat class name from filename
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
        plot_filename = f"evaluation_v{model_version}_{bat_type}_{source}_{background}_{timestamp}.png"
        plot_path = os.path.join(output_dir, plot_filename)

        # Create the plot
        create_siamese_confusion_matrix_plot(
            binary_true_labels,
            binary_pred_labels,
            class_verification_rates,
            plot_path,
            model_version,
            bat_type,
            source,
            background,
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
    # Basic usage - process all bat classes in data folder
    python generate_predictions.py --model /path/to/model --input /path/to/data --model-version 1 --bat-type r --source video --background original
    
    # Process specific bat classes only
    python generate_predictions.py --model /path/to/model --input /path/to/data --model-version 2 --bat-type m --source still --background green --subdirs W H rasmi charlie
    
    # With verbose output
    python generate_predictions.py -m /path/to/model -i /path/to/data --model-version 1 --bat-type r --source video --background original --verbose
    
    # Fast testing with limited pairs
    python generate_predictions.py -m /path/to/model -i /path/to/data --model-version 1 --bat-type r --source video --background random --max-pairs 100
    
    # Using short argument names
    python generate_predictions.py -m model_path -i data_path --model-version 3 --bat-type m --source still --background original
    
Note: All outputs are saved to the 'evaluations/' directory in the project root.
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
        "--model-version",
        type=int,
        required=True,
        help="Model version number (e.g., 1, 2, 3)",
    )

    parser.add_argument(
        "--bat-type",
        required=True,
        choices=["r", "m"],
        help="Bat type: r=Rousettus, m=Mauritius",
    )

    parser.add_argument(
        "--source",
        required=True,
        choices=["video", "still"],
        help="Image source type (video or still images)",
    )

    parser.add_argument(
        "--background",
        required=True,
        choices=["green", "random", "original"],
        help="Background type used (green, random, or original)",
    )

    parser.add_argument(
        "--subdirs",
        nargs="*",
        help="Specific bat classes to process (e.g., W H rasmi charlie)",
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
            model_version=args.model_version,
            bat_type=args.bat_type,
            source=args.source,
            background=args.background,
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

import sys
import click
import os
import shutil
from pathlib import Path
import random

# Add app directory to path for imports
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from utils.filename_parser import group_files_by_class, is_augmented_file


@click.command()
@click.version_option(version="1.0.0")
@click.option("--input", "-i", type=click.Path(exists=True, file_okay=False, dir_okay=True), 
              required=True, help="Input directory containing image files")
@click.option("--train-ratio", "-t", type=float, default=0.7,
              help="Ratio of files for training set per class (default: 0.7)")
@click.option("--seed", "-s", type=int, default=42,
              help="Random seed for reproducibility (default: 42)")
def split_files(input, train_ratio, seed):
    """Split image files from input directory into train and validation sets per class.
    
    This command splits files on a per-class basis to maintain class distribution.
    Files must follow the naming convention: type--class--id[--aug###]
    
    NOTE: Augmented files (--aug###) are automatically filtered out to prevent data leakage.
    Augmented images should be generated AFTER the split, not before, to ensure that
    augmentations of the same base image don't end up in both train and validation sets.
    """
    input_path = Path(input)
    parent_dir = input_path.parent
    
    # Create output directories
    output_train = input_path / "train"
    output_validation = input_path / "validation"
    
    output_train.mkdir(exist_ok=True)
    output_validation.mkdir(exist_ok=True)
    
    # Get all image files from input directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    all_files = [f for f in input_path.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not all_files:
        click.echo(f"No image files found in {input_path}")
        return
    
    # Filter out augmented files to prevent data leakage
    non_augmented_files = []
    augmented_count = 0
    for f in all_files:
        if is_augmented_file(f.name):
            augmented_count += 1
        else:
            non_augmented_files.append(str(f))
    
    if augmented_count > 0:
        click.echo(f"⚠️  Filtered out {augmented_count} augmented files to prevent data leakage")
        click.echo(f"   Augmented images should be generated AFTER splitting")
    
    if not non_augmented_files:
        click.echo("No non-augmented image files found after filtering")
        return
    
    # Group files by class
    class_files = group_files_by_class(non_augmented_files)
    
    if not class_files:
        click.echo("No files with valid naming convention found")
        click.echo("Expected pattern: type--class--id (e.g., img--bat_species_a--001)")
        return
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Split files per class
    click.echo(f"\nSplitting {len(non_augmented_files)} files across {len(class_files)} classes:")
    click.echo(f"  Train ratio: {train_ratio:.1%}")
    click.echo(f"  Validation ratio: {1-train_ratio:.1%}\n")
    
    total_train = 0
    total_validation = 0
    
    for class_name, ids_dict in class_files.items():
        # Get all IDs for this class
        all_ids = list(ids_dict.keys())
        random.shuffle(all_ids)
        
        # Split IDs (not individual files) to prevent leakage
        train_id_count = max(1, int(len(all_ids) * train_ratio))
        train_ids = all_ids[:train_id_count]
        validation_ids = all_ids[train_id_count:]
        
        # Collect files for each split
        train_files_for_class = []
        validation_files_for_class = []
        
        for id in train_ids:
            train_files_for_class.extend(ids_dict[id])
        
        for id in validation_ids:
            validation_files_for_class.extend(ids_dict[id])
        
        # Copy files to respective directories
        for file_path in train_files_for_class:
            file_name = os.path.basename(file_path)
            shutil.copy2(file_path, output_train / file_name)
        
        for file_path in validation_files_for_class:
            file_name = os.path.basename(file_path)
            shutil.copy2(file_path, output_validation / file_name)
        
        total_train += len(train_files_for_class)
        total_validation += len(validation_files_for_class)
        
        click.echo(f"  Class '{class_name}': "
                  f"{len(train_files_for_class)} train, "
                  f"{len(validation_files_for_class)} validation "
                  f"({len(all_ids)} IDs total)")
    
    click.echo(f"\n✓ Files split successfully!")
    click.echo(f"  Total train files: {total_train} -> {output_train}")
    click.echo(f"  Total validation files: {total_validation} -> {output_validation}")


if __name__ == "__main__":
    split_files()
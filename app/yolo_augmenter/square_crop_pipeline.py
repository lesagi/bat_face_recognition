#!/usr/bin/env python3
"""
Square Crop Pipeline for YOLO Segmentation Datasets

This module provides a CLI interface for batch processing multiple datasets
with square cropping around segmentation boundaries.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from yolo_augmenter.square_segmentation_cropper import SquareSegmentationCropper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SquareCropPipeline:
    """Pipeline for processing multiple datasets with square cropping."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def discover_datasets(self, base_dir: str = "data/interim") -> List[Path]:
        """
        Auto-discover available datasets in the base directory.
        
        Args:
            base_dir: Base directory to search for datasets
            
        Returns:
            List of dataset directories that have images+labels structure
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning(f"Base directory does not exist: {base_path}")
            return []
        
        datasets = []
        
        # Look for directories that contain both images and labels subdirectories
        for item in base_path.iterdir():
            if item.is_dir():
                images_dir = item / "images"
                labels_dir = item / "labels"
                
                if images_dir.exists() and labels_dir.exists():
                    # Check if there are actual files
                    image_files = list(images_dir.glob("*"))
                    label_files = list(labels_dir.glob("*.txt"))
                    
                    if image_files and label_files:
                        datasets.append(item)
                        logger.info(f"Found dataset: {item.name} ({len(image_files)} images, {len(label_files)} labels)")
        
        return datasets
    
    def process_dataset(self, 
                       input_dir: str, 
                       output_dir: str, 
                       buffer_multiplier: float = 1.1,
                       output_format: str = "jpg",
                       quality: int = 95) -> Dict[str, Any]:
        """
        Process a single dataset.
        
        Args:
            input_dir: Input dataset directory
            output_dir: Output directory for processed data
            buffer_multiplier: Buffer around segmentation
            output_format: Output image format
            quality: Image quality for lossy formats
            
        Returns:
            Processing results dictionary
        """
        try:
            logger.info(f"Processing dataset: {input_dir}")
            
            cropper = SquareSegmentationCropper(
                input_dir=input_dir,
                output_dir=output_dir,
                buffer_multiplier=buffer_multiplier,
                output_format=output_format,
                quality=quality,
                verbose=self.verbose
            )
            
            results = cropper.process_dataset()
            logger.info(f"Dataset {input_dir} processing complete: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing dataset {input_dir}: {e}")
            return {
                "success": False,
                "error": str(e),
                "input_dir": input_dir
            }
    
    def process_all_datasets(self, 
                           base_dir: str = "data/interim",
                           output_base_dir: str = "data/processed",
                           buffer_multiplier: float = 1.1,
                           output_format: str = "jpg",
                           quality: int = 95) -> Dict[str, Any]:
        """
        Process all discovered datasets.
        
        Args:
            base_dir: Base directory to search for datasets
            output_base_dir: Base directory for output
            buffer_multiplier: Buffer around segmentation
            output_format: Output image format
            quality: Image quality for lossy formats
            
        Returns:
            Summary of all processing results
        """
        logger.info("Starting batch processing of all datasets...")
        
        # Discover datasets
        datasets = self.discover_datasets(base_dir)
        
        if not datasets:
            logger.warning("No datasets found for processing")
            return {
                "success": False,
                "error": "No datasets found",
                "total_datasets": 0
            }
        
        logger.info(f"Found {len(datasets)} datasets to process")
        
        # Process each dataset
        results = []
        successful = 0
        failed = 0
        
        for dataset_dir in datasets:
            dataset_name = dataset_dir.name
            output_dir = Path(output_base_dir) / f"{dataset_name}_square_crops"
            
            logger.info(f"Processing dataset: {dataset_name}")
            
            result = self.process_dataset(
                input_dir=str(dataset_dir),
                output_dir=str(output_dir),
                buffer_multiplier=buffer_multiplier,
                output_format=output_format,
                quality=quality
            )
            
            results.append({
                "dataset": dataset_name,
                "result": result
            })
            
            if result.get("success", False):
                successful += 1
            else:
                failed += 1
        
        # Summary
        summary = {
            "success": failed == 0,
            "total_datasets": len(datasets),
            "successful": successful,
            "failed": failed,
            "results": results
        }
        
        logger.info("Batch processing complete!")
        logger.info(f"Total datasets: {len(datasets)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        
        return summary


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Square Crop Pipeline for YOLO Segmentation Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets with defaults
  python -m app.yolo_augmenter.square_crop_pipeline
  
  # Process specific directory
  python -m app.yolo_augmenter.square_crop_pipeline \\
      --input-dir data/interim/mauritius_images+labels \\
      --output-dir data/processed/mauritius_square_crops
  
  # Custom parameters
  python -m app.yolo_augmenter.square_crop_pipeline \\
      --buffer-multiplier 1.2 \\
      --output-format png \\
      --quality 90
  
  # Process all datasets with custom settings
  python -m app.yolo_augmenter.square_crop_pipeline \\
      --all-datasets \\
      --buffer-multiplier 1.1 \\
      --output-format jpg
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "-i", "--input-dir",
        help="Input directory containing images and labels (default: auto-detect from config)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for processed data (default: auto-generate in data/processed/)"
    )
    
    # Processing parameters
    parser.add_argument(
        "--buffer-multiplier",
        type=float,
        default=1.1,
        help="Buffer around segmentation boundaries (default: 1.1)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["jpg", "jpeg", "png", "bmp", "tiff"],
        default="jpg",
        help="Output image format (default: jpg)"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Image quality for lossy formats (default: 95)"
    )
    
    # Dataset processing options
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Process all available datasets (default: True)"
    )
    
    parser.add_argument(
        "--base-dir",
        default="data/interim",
        help="Base directory to search for datasets (default: data/interim)"
    )
    
    parser.add_argument(
        "--output-base-dir",
        default="data/processed",
        help="Base directory for output (default: data/processed)"
    )
    
    # Control options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing it"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create pipeline
    pipeline = SquareCropPipeline(verbose=args.verbose)
    
    try:
        if args.dry_run:
            # Show what would be processed
            logger.info("DRY RUN MODE - No actual processing will occur")
            
            if args.input_dir:
                logger.info(f"Would process single directory: {args.input_dir}")
                if args.output_dir:
                    logger.info(f"Output would go to: {args.output_dir}")
                else:
                    logger.info("Output directory would be auto-generated")
            else:
                datasets = pipeline.discover_datasets(args.base_dir)
                logger.info(f"Would process {len(datasets)} datasets:")
                for dataset in datasets:
                    output_dir = Path(args.output_base_dir) / f"{dataset.name}_square_crops"
                    logger.info(f"  {dataset.name} -> {output_dir}")
            
            logger.info("Dry run complete. Use --help for more options.")
            return 0
        
        # Process datasets
        if args.input_dir:
            # Process single directory
            if not args.output_dir:
                # Auto-generate output directory
                input_path = Path(args.input_dir)
                args.output_dir = f"data/processed/{input_path.name}_square_crops"
            
            logger.info(f"Processing single directory: {args.input_dir}")
            result = pipeline.process_dataset(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                buffer_multiplier=args.buffer_multiplier,
                output_format=args.output_format,
                quality=args.quality
            )
            
            if result.get("success", False):
                logger.info("Single directory processing completed successfully!")
                return 0
            else:
                logger.error(f"Single directory processing failed: {result.get('error', 'Unknown error')}")
                return 1
        
        else:
            # Process all datasets
            logger.info("Processing all available datasets...")
            result = pipeline.process_all_datasets(
                base_dir=args.base_dir,
                output_base_dir=args.output_base_dir,
                buffer_multiplier=args.buffer_multiplier,
                output_format=args.output_format,
                quality=args.quality
            )
            
            if result.get("success", False):
                logger.info("All datasets processed successfully!")
                return 0
            else:
                logger.error(f"Some datasets failed to process: {result.get('failed', 0)} failures")
                return 1
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

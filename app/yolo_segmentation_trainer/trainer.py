"""
YOLO Segmentation Trainer for bat face segmentation models.

This module provides a trainer class for training YOLO segmentation models.
"""

import os
from typing import Optional
from pathlib import Path
from ultralytics import YOLO


class YoloSegmentationTrainer:
    """Trainer class for YOLO segmentation models."""
    
    def __init__(self, 
                 training_data_path: str,
                 epochs: int = 80,
                 base_model: str = 'yolov8n-seg.pt'):
        """Initialize the YOLO segmentation trainer.
        
        Args:
            config_path: Path to the YOLO dataset configuration file
            epochs: Number of training epochs
            base_model: Base YOLO model to use for training
            
        Raises:
            FileNotFoundError: If config_path doesn't exist
            ValueError: If epochs is not positive
        """
        if not os.path.exists(training_data_path):
            raise FileNotFoundError(f"Config file not found: {training_data_path}")
        
        if epochs <= 0:
            raise ValueError(f"Epochs must be positive, got {epochs}")
            
        self.training_data_path = training_data_path
        self.epochs = epochs
        self.base_model = base_model
        self.model = None
    
    def train(self, 
              save_dir: Optional[str] = None,
              patience: int = 50,
              batch_size: int = 16,
              image_size: int = 640,
              workers: int = 8) -> str:
        """Train the YOLO segmentation model.
        
        Args:
            save_dir: Directory to save training results (optional)
            patience: Early stopping patience
            batch_size: Training batch size
            image_size: Input image size
            workers: Number of data loading workers
            
        Returns:
            Path to the trained model
            
        Raises:
            RuntimeError: If training fails
        """
        if save_dir is None:
            save_dir = os.path.join(Path(self.training_data_path).parent, "training_results")
        
        try:
            # Load the base model
            self.model = YOLO(self.base_model)
            print(f"Loaded base model: {self.base_model}")
            
            # Start training
            print(f"Starting training for {self.epochs} epochs...")
            results = self.model.train(
                data=self.training_data_path,
                epochs=self.epochs,
                patience=patience,
                batch=batch_size,
                imgsz=image_size,
                workers=workers,
                project=save_dir
            )
            
            # Get the path to the best model
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            print(f"Training completed. Best model saved to: {best_model_path}")
            
            return str(best_model_path)
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}") 
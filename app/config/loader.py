"""
General Configuration Loader for Bat Face Recognition Project

This module provides a centralized configuration loader that loads all project
configuration from app/config/config.yml and provides defined getters for all access.
"""

import yaml
from typing import Dict, Any
from pathlib import Path

from .general import GeneralConfig
from .mlflow import MLflowConfig
from .models import ModelsConfig
from .siamese_network import SiameseNetworkConfig
from .yolo_segmentation import YOLOSegmentationConfig
from .video_processing import VideoProcessingConfig
from .visualization import VisualizationConfig


class ConfigLoader:
    def __init__(self):
        self.config_path = Path(__file__).parent / "config.yml"
        self.config = self._load_config()

        self.general = GeneralConfig(self.config.get("general", {}))
        self.mlflow = MLflowConfig(self.config.get("mlflow", {}))
        self.models = ModelsConfig(self.config.get("models", {}))
        self.siamese_network = SiameseNetworkConfig(self.config.get("siamese_network", {}))
        self.yolo_segmentation = YOLOSegmentationConfig(self.config.get("yolo_segmentation", {}))
        self.video_processing = VideoProcessingConfig(self.config.get("video_processing", {}))
        self.visualization = VisualizationConfig(self.config.get("visualization", {}))

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        return config or {}

    def get_full_config(self) -> Dict[str, Any]:
        return self.config

    def print_config_summary(self):
        print("=" * 80)
        print("BAT FACE RECOGNITION PROJECT - CONFIGURATION SUMMARY")
        print("=" * 80)

        print(f"\n🔧 GENERAL SETTINGS:")
        print(f"   GPU Enabled: {self.general.gpu_enabled}")
        print(f"   Random Seed: {self.general.random_seed}")

        print(f"\n📈 MLFLOW:")
        print(f"   Enabled: {self.mlflow.enabled}")
        print(f"   Experiment: {self.mlflow.experiment_name}")
        print(f"   Tracking Host: {self.mlflow.tracking_host}")
        print(f"   Tracking Port: {self.mlflow.tracking_port}")
        print(f"   Tracking URI: {self.mlflow.tracking_uri}")

        if self.models._config:
            print(f"\n🤖 MODELS:")
            for model_name, model_config in self.models._config.items():
                print(f"   {model_name.title()}: {model_config.get('model_path', 'N/A')}")
                print(f"     Type: {model_config.get('model_type', 'N/A')}")
                print(f"     Confidence: {model_config.get('confidence_threshold', 'N/A')}")

        if self.siamese_network._config:
            training_config = self.siamese_network.training
            model_config = self.siamese_network.model

            print(f"\n🧠 SIAMESE NETWORK:")
            print(f"   Input Size: {model_config.get('input_edge_length', 'N/A')}x{model_config.get('input_edge_length', 'N/A')}")
            print(f"   Epochs: {training_config.get('epochs', 'N/A')}")
            print(f"   Batch Size: {training_config.get('batch_size', 'N/A')}")
            print(f"   Learning Rate: {training_config.get('learning_rate', 'N/A')}")

        if self.video_processing._config:
            print(f"\n🎥 VIDEO PROCESSING:")
            print(f"   Processing Limit: {self.video_processing.processing_limit}")

        if self.yolo_segmentation._config:
            training_config = self.yolo_segmentation.training
            print(f"\n🎯 YOLO SEGMENTATION:")
            print(f"   Default Epochs: {training_config.get('default_epochs', 'N/A')}")
            print(f"   Default Batch Size: {training_config.get('default_batch_size', 'N/A')}")
            print(f"   Default Confidence: {training_config.get('default_confidence', 'N/A')}")

        print("=" * 80)


def load_config() -> ConfigLoader:
    return ConfigLoader()


if __name__ == "__main__":
    config_loader = load_config()

    config_loader.print_config_summary()
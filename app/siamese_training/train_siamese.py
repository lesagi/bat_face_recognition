#!/usr/bin/env python3
"""
Simple script to train the Siamese Network
"""

import os
import sys
import argparse

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
app_dir = os.path.dirname(parent_dir)
sys.path.insert(0, app_dir)

from app.siamese_training.trainer import SiameseNetworkTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Siamese Network")
    parser.add_argument("--bat-type", dest="bat_type", type=str, required=True, choices=['m', 'r'],
                        help="Bat type: 'm' for mauritius, 'r' for rousettus")
    parser.add_argument("--augmented-data", dest="augmented_data", action="store_true",
                        help="Whether the data is augmented")
    parser.add_argument("--data-source", dest="data_source", type=str, required=True, choices=['video', 'still'],
                        help="Data source: 'video' or 'still'")
    parser.add_argument("--background", dest="background", type=str, required=True,
                        choices=["green", "random", "original"],
                        help="Background type: 'green', 'random', or 'original'")
    parser.add_argument("--split-mode", dest="split_mode", type=str, required=True,
                        choices=["image_split", "bat_split"],
                        help="Train/test split: 'image_split' (per-class images) or 'bat_split' (disjoint identities)")
    parser.add_argument("--permute-labels", dest="permute_labels", action="store_true",
                        help="Enable label permutation test mode")
    args = parser.parse_args()

    print("🚀 Starting Siamese Network Training")
    print(f"📁 Background type: {args.background}")
    print(f"✂️  Train/test split mode: {args.split_mode}")
    print()

    try:
        # Create and run the trainer
        print("🔧 Initializing trainer...")
        trainer = SiameseNetworkTrainer(
            args.bat_type, args.augmented_data, args.data_source,
            background=args.background,
            split_mode=args.split_mode,
            permute_labels=args.permute_labels,
        )
        
        print("✅ Trainer initialized successfully!")
        print(f"📊 Training pairs: {trainer.train_batches.cardinality().numpy()}")
        print(f"📊 Testing pairs: {trainer.test_batches.cardinality().numpy()}")
        print(f"🔧 Batch size: {trainer.batch_size}")
        print(f"🔄 Epochs: {trainer.num_epochs}")
        print()
        
        print("🎯 Starting training...")
        trainer.fit(args.bat_type, args.augmented_data, args.data_source)

        print()
        print("🎉 Training completed successfully!")
        print(f"📊 Model saved to: {trainer.model_output_dir}")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

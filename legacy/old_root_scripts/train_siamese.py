#!/usr/bin/env python3
"""
Simple script to train the Siamese Network
"""

import sys
import os

# Add the app directory to the path
sys.path.append("app")

from siamese_network.trainer import SiameseNetworkTrainer


def main():
    # Input directory with processed images
    input_dir = "/home/sagilevi1/bat_face_rec_project/face_rec_rousettus_#1/background_replacement/processed_siamese_picsum"

    print("🚀 Starting Siamese Network Training")
    print(f"📁 Input directory: {input_dir}")
    print("⏳ This may take a while (80 epochs)...")
    print()

    try:
        # Create and run the trainer
        print("📋 Initializing trainer...")
        trainer = SiameseNetworkTrainer(input_dir)

        print("✅ Trainer initialized successfully!")
        print(f"   - Train batches: {trainer.train_batches.cardinality().numpy()}")
        print(f"   - Test batches: {trainer.test_batches.cardinality().numpy()}")
        print()

        print("🚀 Starting training...")
        trainer.fit()

        print()
        print("🎉 Training completed successfully!")
        print("📊 Check the logs and saved models in the runs directory")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

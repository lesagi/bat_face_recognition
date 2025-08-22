#!/usr/bin/env python3
"""
Simple script to train the Siamese Network
"""

from trainer import SiameseNetworkTrainer


def main():
    # Input directory with processed images
    input_dir = "/Users/MAC/Documents/bat_face_rec/face_rec_rousettus_#1/background_replacement/processed_siamese_picsum"

    print("🚀 Starting Siamese Network Training")
    print(f"📁 Input directory: {input_dir}")
    print()

    try:
        # Create and run the trainer
        trainer = SiameseNetworkTrainer(input_dir)
        trainer.fit()

        print()
        print("🎉 Training completed successfully!")
        print(
            f"📊 Model saved to: {trainer.log_file_path.replace('training_log.csv', '')}"
        )

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

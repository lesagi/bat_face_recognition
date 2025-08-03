import os
import cv2
from utils.image_utils import strip_filename_from_path
from background_replacement.processor import BackgroundReplacementProcessor


class VideoImagesExtractor:
    def __init__(self, video_path, output_base_dir):
        self.video_path = video_path
        self.output_base_dir = output_base_dir
        self.image_processor = BackgroundReplacementProcessor()

    def process_video(
        self,
        preprocess_frame=None,
        post_prediction_processor=None,
        output_sub_dir_name=None,
        threshold=0.5,
    ):
        video_name = os.path.basename(self.video_path)
        output_folder = os.path.join(
            self.output_base_dir, os.path.splitext(video_name)[0]
        )
        if output_sub_dir_name:
            output_folder = os.path.join(output_folder, output_sub_dir_name)

        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            serial_filename = f"{strip_filename_from_path(video_name)}.{frame_number}"

            frame = self.image_processor.process(
                frame,
                preprocess_frame,
                process_frame_of_masks_prediction,
                post_prediction_processor,
                threshold,
            )
            cv2.imwrite(os.path.join(output_folder, f"{serial_filename}.jpg"), frame)

        cap.release()


def extract_images_from_video(post_process=crop_rectangle_from_cv2_frame):
    for video_file in os.listdir(VIDEOS_DIR)[:VIDEOS_PROCESSING_LIMIT]:
        video_image_extractor = VideoImagesExtractor(
            os.path.join(VIDEOS_DIR, video_file), PROCESSED_VIDEOS_BASE_DIR
        )
        video_image_extractor.process_video(
            post_prediction_processor=post_process,
            output_sub_dir_name=PROCESSED_VIDEOS_SUB_DIR,
            threshold=0.8,
        )

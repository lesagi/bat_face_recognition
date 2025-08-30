import random
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
# Try absolute imports first (when run as module), fall back to relative imports (when run directly)
try:
    from app.utils.image_utils import strip_filename_from_path
except ImportError:
    # Fallback to relative imports when running script directly
    # Add parent directory to path for relative imports
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from utils.image_utils import strip_filename_from_path
import os
import time
from requests.exceptions import RequestException, Timeout, ConnectionError


def create_blur_image(height, width):
    # Create a random image (each pixel has random RGB values)
    random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Choose a random odd kernel size between 3 and 29 (Gaussian blur requires odd dimensions)
    kernel_size = random.choice([k for k in range(3, 30) if k % 2 == 1])

    # Apply Gaussian Blur with the random kernel size
    blurry_image = cv2.GaussianBlur(random_image, (kernel_size, kernel_size), 0)
    return blurry_image


def create_green_image(height, width):
    # Create a new image with the specified dimensions and 3 color channels
    green_image = np.zeros((height, width, 3), dtype=np.uint8)
    green_image[:] = (
        0,
        255,
        0,
    )  # In OpenCV, this represents green (BGR order: Blue=0, Green=255, Red=0)
    return green_image


def process_frame_of_masks_prediction(frame, mask):
    frame_height, frame_width = frame.shape[:2]
    background_image = create_blur_image(height=frame_height, width=frame_width)

    # Choose a random odd kernel size between 3 and 29 (Gaussian blur requires odd dimensions)
    kernel_size = random.choice([k for k in range(3, 30) if k % 2 == 1])

    # Apply Gaussian Blur with the random kernel size
    blurry_image = cv2.GaussianBlur(background_image, (kernel_size, kernel_size), 0)

    # Convert the mask to a numpy array
    mask = mask.cpu().numpy()
    mask = cv2.resize(mask, (frame_width, frame_height))

    # Convert mask to boolean array
    mask = mask.astype(bool)

    # Repeat the mask along the color dimension
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Use the mask to extract ROI from frame
    roi = frame * mask

    # Use the mask to replace the corresponding region in the green image with the ROI
    blurry_image[mask] = roi[mask]

    return blurry_image


def get_random_cropped_image(
    height, width, max_retries=3, timeout=30, fallback_to_generated=True
):
    """
    Fetches a random image from Picsum Photos and crops it to the given height and width.
    Includes retry mechanism with exponential backoff for robust network handling.

    Parameters:
        height (int): The desired height of the cropped image.
        width (int): The desired width of the cropped image.
        max_retries (int): Maximum number of retry attempts (default: 3)
        timeout (float): Timeout for each request in seconds (default: 10)
        fallback_to_generated (bool): If True, generates a fallback image when all retries fail

    Returns:
        image_np (numpy.ndarray): The cropped image in BGR format (suitable for OpenCV).

    Raises:
        Exception: If all retries fail and fallback_to_generated is False
    """
    # Ensure we fetch an image that's at least as large as the desired dimensions.
    # We take the max to get a square image that will definitely cover both dimensions.
    min_dim = max(height, width)
    url = f"https://picsum.photos/{min_dim}/{min_dim}"

    last_exception = None

    for attempt in range(
        max_retries + 1
    ):  # +1 because we want max_retries actual retries
        try:
            print(f"Fetching Picsum image (attempt {attempt + 1}/{max_retries + 1})...")

            # Make request with timeout
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                # Success! Process the image
                image = Image.open(BytesIO(response.content))

                # Calculate the coordinates to crop the image to the desired size (center crop)
                img_width, img_height = image.size
                left = (img_width - width) // 2
                top = (img_height - height) // 2
                right = left + width
                bottom = top + height

                cropped_image = image.crop((left, top, right, bottom))

                # Convert the cropped image to a NumPy array and convert from RGB to BGR (for OpenCV)
                image_np = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
                print(f"✓ Successfully fetched Picsum image on attempt {attempt + 1}")
                return image_np
            else:
                raise Exception(
                    f"HTTP {response.status_code}: Failed to fetch image from Picsum"
                )

        except (RequestException, ConnectionError, Timeout) as e:
            last_exception = e
            print(f"✗ Network error on attempt {attempt + 1}: {type(e).__name__}: {e}")

        except Exception as e:
            last_exception = e
            print(f"✗ Error on attempt {attempt + 1}: {type(e).__name__}: {e}")

        # Don't sleep after the last attempt
        if attempt < max_retries:
            # Exponential backoff: 1s, 2s, 4s, 8s...
            sleep_time = 2**attempt
            print(f"⏳ Waiting {sleep_time}s before retry...")
            time.sleep(sleep_time)

    # All retries failed
    print(f"✗ All {max_retries + 1} attempts failed to fetch Picsum image")

    if fallback_to_generated:
        print("🎨 Generating fallback gradient background...")
        return _generate_fallback_background(height, width)
    else:
        raise Exception(
            f"Failed to fetch image from Picsum after {max_retries + 1} attempts. Last error: {last_exception}"
        )


def _generate_fallback_background(height, width):
    """
    Generate a fallback background when Picsum fails.
    Creates a gradient from random colors.

    Parameters:
        height (int): Height of the background
        width (int): Width of the background

    Returns:
        numpy.ndarray: Generated background in BGR format
    """
    # Generate random gradient colors
    start_color = tuple(random.randint(50, 150) for _ in range(3))  # Darker colors
    end_color = tuple(random.randint(150, 255) for _ in range(3))  # Lighter colors

    # Create gradient background
    background = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        # Linear interpolation between start and end colors
        ratio = i / height
        current_color = tuple(
            int(start_color[j] * (1 - ratio) + end_color[j] * ratio) for j in range(3)
        )
        background[i, :] = current_color

    return background


def replace_green_background(image):
    """
    Replaces the green background in an image with a random image from Picsum Photos.

    Parameters:
        image (numpy.ndarray): Image containing an object with a green background (BGR format).

    Returns:
        result (numpy.ndarray): The image with the green background replaced.
    """
    # Get dimensions of the input image
    height, width = image.shape[:2]

    # Fetch a random background image cropped to the same dimensions
    random_background = get_random_cropped_image(height, width)

    # Define lower and upper bounds for the green color (in BGR)
    # Here, we allow some tolerance around (0, 255, 0)
    lower_green = np.array([0, 230, 0], dtype=np.uint8)
    upper_green = np.array([30, 255, 30], dtype=np.uint8)

    # Create a mask for pixels within the green range
    green_mask = cv2.inRange(image, lower_green, upper_green)

    # Optionally, you can smooth the mask if needed:
    green_mask = cv2.medianBlur(green_mask, 5)

    # Invert mask to get the object area (non-green regions)
    object_mask = cv2.bitwise_not(green_mask)

    # Extract the object from the original image using the inverted mask
    object_foreground = cv2.bitwise_and(image, image, mask=object_mask)

    # Extract the background from the random image using the green mask
    background_region = cv2.bitwise_and(
        random_background, random_background, mask=green_mask
    )

    # Combine the object and the new background
    result = cv2.add(object_foreground, background_region)
    return result


def replace_green_background_to_bat_face_image(image_path, output_path):
    # Load the image from a file path
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    image = replace_green_background(image)

    filename = strip_filename_from_path(image_path)
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(f"{os.path.join(output_path, filename)}.png", image)
    cv2.destroyAllWindows()


def preprocess_siamese_input(image_path, target_size=(224, 224)):
    """
    Preprocess an image for input to the siamese network.

    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the image (height, width)

    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Resize the image
    img = cv2.resize(img, target_size)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img

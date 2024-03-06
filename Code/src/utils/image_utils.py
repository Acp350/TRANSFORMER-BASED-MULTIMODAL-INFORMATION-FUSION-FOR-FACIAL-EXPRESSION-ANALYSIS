import numpy as np
import cv2
import os
from PIL import Image

def process_image(image_path, output_path, new_size=(48, 48), kernel_size=2, dilation_iterations=2, softening_threshold=38):
    """
    Process an image: Convert to grayscale, resize, apply dilation, and soften.
    """
    # Open image
    img = Image.open(image_path)

    # Convert to grayscale
    img_gray = img.convert('LA')

    # Resize image
    img_resized = img_gray.resize(new_size)

    # Convert to numpy array for cv2 operations
    img_array = np.array(img_resized)

    # Create dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8) * 0.5
    img_dilated = cv2.dilate(img_array, kernel, iterations=dilation_iterations)

    # Soften image
    img_dilated[img_dilated >= softening_threshold] = 255  # white
    img_dilated[img_dilated < softening_threshold] = 127   # gray

    # Save processed image
    os.makedirs(output_path, exist_ok=True)
    Image.fromarray(img_dilated).save(os.path.join(output_path, os.path.basename(image_path).rsplit(".")[0] + ".png"))

def process_images_in_subfolder(in_path_imgs, out_path_imgs, new_size):
    """
    Process all images in a specified subfolder.
    """
    for img_subfolder in os.listdir(in_path_imgs):
        in_path_subfolder = os.path.join(in_path_imgs, img_subfolder)
        out_path_subfolder = os.path.join(out_path_imgs, img_subfolder)
        if not os.path.isdir(in_path_subfolder):
            continue
        for img_name in os.listdir(in_path_subfolder):
            img_path = os.path.join(in_path_subfolder, img_name)
            output_path = os.path.join(out_path_subfolder, img_name.rsplit(".")[0] + ".png")
            if os.path.exists(output_path):
                continue
            try:
                process_image(img_path, out_path_subfolder, new_size=new_size)
            except Exception as e:
                print(f"Problems with IMAGE: {img_path}, Error: {e}")

# Example usage
in_path = 'path_to_input_images'
out_path = 'path_to_output_images'
new_size = (48, 48)
process_images_in_subfolder(in_path, out_path, new_size)

import numpy as np
from PIL import Image


# qt libraries
from qtpy.QtWidgets import *

# app specific libraries
import control.gui as gui

if __name__ == "__main__":
    # Define the file path and image properties
    raw_file_path = "F0000000.raw"  # Path to your .raw file
    output_image_path = "output.bmp"  # Path to save the output image
    width = 1024  # Image width
    height = 608  # Image height
    bit_depth = 8  # Bits per pixel (8 for grayscale, 24 for RGB)

    # Load the .raw file
    with open(raw_file_path, 'rb') as raw_file:
        raw_data = raw_file.read()

    # Convert the raw data into a numpy array
    if bit_depth == 8:
        image_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
    elif bit_depth == 24:
        image_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
    else:
        raise ValueError("Unsupported bit depth")

    # Convert the numpy array to an image
    image = Image.fromarray(image_array)

    # Save the image
    image.save(output_image_path)

    print(f"Image saved to {output_image_path}")

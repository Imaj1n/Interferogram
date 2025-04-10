import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import pandas as pd

def read_img_as_interferogram(filename,row):
    try:
        image = io.imread(filename)
    except Exception as e:
        print(f"Error loading image: {e}")

    # Convert to RGB if grayscale
    if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)

    # Display image dimensions
    height, width = image.shape[:2]
    print(f"\nImage loaded: {width} x {height} pixels")



    # Create figure
    plt.figure(figsize=(12, 8))
    plt.title(f"Interferogram Intensity Distribution at Row {row + 1}') ")

            # # Get row input
            # while True:
            #     try:
            #         row = int(input(f"\nEnter row number (1-{height}): ")) - 1
            #         if 0 <= row < height:
            #             break
            #         print(f"Please enter a value between 1 and {height}")
            #     except ValueError:
            #         print("Please enter a valid integer.")

            # Extract and process the row data
    row_data = image[row, :, :]

    # Convert to grayscale using standard weights
    grayscale = np.dot(row_data[..., :3], [0.2989, 0.5870, 0.1140])

    # Normalize the data
    normalized = (grayscale - np.min(grayscale)) / (np.max(grayscale) - np.min(grayscale))

    # Plot the intensity distribution
    # plt.subplot(2, 2, (2, 4))
    plt.plot(normalized)
    plt.xlim(0, len(normalized))
    plt.ylim(0, 1)
    plt.xlabel('Pixel Position')
    plt.ylabel('Normalized Intensity')
    plt.title(f'Intensity Distribution at Row {row + 1}')
    return plt

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

    # Konversi RGB ke grayscale
    if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)

    # Display dimensi foto
    height, width = image.shape[:2]
    print(f"\nImage loaded: {width} x {height} pixels")

    plt.figure(figsize=(12, 8))
    plt.title(f"Interferogram Intensity Distribution at Row {row + 1}') ")

    row_data = image[row, :, :]

    # Konversi grayscale
    grayscale = np.dot(row_data[..., :3], [0.2989, 0.5870, 0.1140])

    # Normalsasi data
    normalized = (grayscale - np.min(grayscale)) / (np.max(grayscale) - np.min(grayscale))

    plt.plot(normalized)
    plt.xlim(0, len(normalized))
    plt.ylim(0, 1)
    plt.xlabel('Intensity Distribution')
    plt.ylabel('Normalized Intensity')
    plt.grid()
    return plt

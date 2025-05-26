import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import streamlit as st
from scipy.signal import savgol_filter
import pywt
import pandas as pd
import os


#fungsi untuk memperhalus noise
def wavelet_denoising(intensity_data, wavelet='db4', level=3):
    coeffs = pywt.wavedec(intensity_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(intensity_data)))
    coeffs[1:] = (pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:])
    return pywt.waverec(coeffs, wavelet)

#fungsi untuk memperhalus noise
def savitzky_golay(intensity_data, window_length=15, polyorder=2):
    return savgol_filter(intensity_data, window_length, polyorder)

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
    if st.checkbox("Tekan ini untuk memperhalus noise"):
      normalized = savitzky_golay(normalized)
      normalized = wavelet_denoising(normalized)
    else : 
      pass

    plt.figure(figsize=(10.5,4.5))
    plt.plot(normalized)
    plt.xlim(0, len(normalized))
    plt.ylim(0, 1)
    plt.xlabel(f'Pixel Position')
    plt.ylabel('Normalized Intensity Distribution')
    plt.grid()

    data = {
    'Intensitas': normalized
    }
    df = pd.DataFrame(data)


    return plt,df

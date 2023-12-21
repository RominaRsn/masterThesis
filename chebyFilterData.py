import tensorflow as tf
import keras
from keras import layers, Input, Model
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch, convolve, cheby2, sosfiltfilt
from masterThesis.metrics import metrics
from keras.utils import plot_model
import os
import masterThesis.model as models
from keras.callbacks import ModelCheckpoint
from scipy.ndimage import convolve1d
import neurokit2 as nk

def cheby2_lowpass(data):
    order = 10
    stopband_attenuation = 40  # in dB
    f1 = 0.1  # Lower stopband frequency in Hz
    f2 = 48.0  # Upper stopband frequency in Hz
    fs = 500  # Sampling frequency in Hz
    #
    # # Design the Chebyshev Type-II IIR bandpass filter
    sos = cheby2(N=order, rs=stopband_attenuation, Wn=[f1/ (0.5 * fs), f2/ (0.5 * fs)], btype='band', analog=False, output='sos')
    # filtered = sosfiltfilt(sos, data)
    # chunk_size = 900
    # num_chunks = len(data) // chunk_size
    #
    # filtered_data = np.empty_like(data)
    #
    # for i in range(num_chunks):
    #     start_idx = i * chunk_size
    #     end_idx = (i + 1) * chunk_size
    #     chunk = data[start_idx:end_idx]
    #
    #     # Apply the filter to the chunk
    #     filtered_chunk = sosfiltfilt(sos, chunk)
    #
    #     # Store the filtered chunk in the result array
    #     filtered_data[start_idx:end_idx] = filtered_chunk
    filtered_data = np.empty_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = sosfiltfilt(sos, data[i, :])
        print(f"filtering {i}th row of data")


    return filtered_data


data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")

data_clean_normalized_filtered = cheby2_lowpass(data_clean_normalized)
np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy", data_clean_normalized_filtered)
data_noisy_normalized_filtered = cheby2_lowpass(data_noisy_normalized)
np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy", data_noisy_normalized_filtered)

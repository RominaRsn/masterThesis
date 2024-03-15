from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch, cheby2, sosfiltfilt
import metrics
from keras.utils import plot_model
import os
import model
import spicy
import tensorflow as tf
import pickle
import masterThesis.metrics as metrics
import neurokit2 as nk
from keras.callbacks import ModelCheckpoint
#
# def cheby2_lowpass(data):
#     order = 10
#     stopband_attenuation = 40  # in dB
#     f1 = 0.1  # Lower stopband frequency in Hz
#     f2 = 48.0  # Upper stopband frequency in Hz
#     fs = 500  # Sampling frequency in Hz
#     #
#     # # Design the Chebyshev Type-II IIR bandpass filter
#     sos = cheby2(N=order, rs=stopband_attenuation, Wn=[f1/ (0.5 * fs), f2/ (0.5 * fs)], btype='band', analog=False, output='sos')
#     # filtered = sosfiltfilt(sos, data)
#     # chunk_size = 900
#     # num_chunks = len(data) // chunk_size
#     #
#     # filtered_data = np.empty_like(data)
#     #
#     # for i in range(num_chunks):
#     #     start_idx = i * chunk_size
#     #     end_idx = (i + 1) * chunk_size
#     #     chunk = data[start_idx:end_idx]
#     #
#     #     # Apply the filter to the chunk
#     #     filtered_chunk = sosfiltfilt(sos, chunk)
#     #
#     #     # Store the filtered chunk in the result array
#     #     filtered_data[start_idx:end_idx] = filtered_chunk
#     filtered_data = np.empty_like(data)
#     for i in range(data.shape[0]):
#         filtered_data[i, :] = sosfiltfilt(sos, data[i, :])
#         print(f"filtering {i}th row of data")
#
#
#     return filtered_data
#
#
# data_clean = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog.npy")
# data_clean = cheby2_lowpass(data_clean)
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_cheby.npy", data_clean)
#
# eog = np.load(r"C:\Users\RominaRsn\Downloads\EEGdenoiseNet-master\data\EOG_all_epochs.npy")
#
# filtered_eog = []
# for i in range(0, eog.shape[0]):
#     filtered_eog.append(nk.signal_filter(eog[i, :], lowcut=25, highcut=120, sampling_rate=256))
#
# filtered_eog = np.array(filtered_eog)
#
# clean_std = np.std(data_clean)
# std_eog = np.std(filtered_eog, axis=1)
#
# scaled = []
# for i in range(0, filtered_eog.shape[0]):
#     scaled.append(filtered_eog[i, :] * (clean_std / std_eog[i]))
#
# scaled = np.array(scaled)
#
# resampled = []
# for i in range(0, filtered_eog.shape[0]):
#     resampled.append(nk.signal_resample(scaled[i, :], sampling_rate=256, desired_sampling_rate=250))
#
# resampled = np.array(resampled)
#
# data_noisy = np.zeros(data_clean.shape)
# for i in range(0, data_clean.shape[0]):
#     multiplicant = np.random.randint(2,5)
#     index = np.random.randint(0, resampled.shape[0])
#     data_noisy[i, :] = data_clean[i, :] + (resampled[index, :] * multiplicant)
#
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_cheby.npy", data_noisy)
#
# #normalize data
# max_clean = np.max(data_clean)
# max_noisy = np.max(data_noisy)
# min_clean = np.min(data_clean)
# min_noisy = np.min(data_noisy)
# max_val = max(max_clean, max_noisy)
# min_val = min(min_clean, min_noisy)
#
# data_clean_normalized = (data_clean - min_val) / (max_val - min_val)
# data_noisy_normalized = (data_noisy - min_val) / (max_val - min_val)
# avg = (np.mean(data_clean_normalized) + np.mean(data_noisy_normalized))/2
#
# data_clean_normalized = data_clean_normalized - avg
# data_noisy_normalized = data_noisy_normalized - avg
#
#
# #train model
# noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)
#
# model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#
# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_checkPoint.h5'
#
# checkpoint = ModelCheckpoint(checkpoint_path,
#                              monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
#                              save_best_only=True,  # Save only if the validation performance improves
#                              mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
#                              verbose=1)
#
# model.optimizer.learning_rate = 1e-6
# model.fit(
#     noisy_train,
#     clean_train,
#     epochs=10,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[callback, checkpoint],
#     shuffle=True
#
# )
# model.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG.h5")
data_clean_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog.npy")
data_noisy_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog.npy")

#normalize eog data
max_clean = np.max(data_clean_eog)
max_noisy = np.max(data_noisy_eog)
min_clean = np.min(data_clean_eog)
min_noisy = np.min(data_noisy_eog)
max_val = max(max_clean, max_noisy)
min_val = min(min_clean, min_noisy)

data_clean_normalized = (data_clean_eog - min_val) / (max_val - min_val)
data_noisy_normalized = (data_noisy_eog - min_val) / (max_val - min_val)
avg = (np.mean(data_clean_normalized) + np.mean(data_noisy_normalized))/2

data_clean_normalized_eog = data_clean_normalized - avg
data_noisy_normalized_eog = data_noisy_normalized - avg

np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_normalized.npy", data_clean_normalized_eog)
np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_normalized.npy", data_noisy_normalized_eog)

# smaller_clean = data_clean_normalized_eog[:, 0:1000]
# smaller_noisy = data_noisy_normalized_eog[:, 0:1000]
#
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_cheby_normalized_smaller.npy", smaller_clean)
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_cheby_normalized_smaller.npy", smaller_noisy)
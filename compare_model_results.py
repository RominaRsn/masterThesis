import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import convolve1d
import neurokit2 as nk
import scipy
from sklearn.metrics import confusion_matrix, recall_score
from collections import Counter
import keras
from keras.models import load_model
from statistics import mean
from sklearn.metrics import f1_score

from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
import masterThesis.model as model
from keras import layers, models, optimizers
import neurokit2 as nk
import cProfile
from memory_profiler import profile


# paths = [r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\realdataCleaning\model_with_three_layers",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\realdataCleaning\model_true_5_layer",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\realdataCleaning\CNN",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\realdataCleaning\GRU",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\realdataCleaning\LSTM"]
#
# fig, axes = plt.subplots(6, 1, sharey='col')
# plt.subplots_adjust(hspace=0.75)
# legend = ["Raw Data", "AE - 3 Layers", "AE - 5 Layers", "CNN Net", "AE GRU", "LSTM Net"]
#
# for index, path in enumerate(paths):
#     result = np.load(os.path.join(path, "pat_1_sz_1_ch_1.npy"))
#     if index == 0:
#         # Normalize data first
#         data = np.load(os.path.join(path, "pat_1_sz_1_ch_1.npy"))
#         mean_val_1 = np.mean(data)
#         std_val_1 = np.std(data)
#         new_normalized_data_1 = (data - mean_val_1) / std_val_1
#         new_normalized_data_1 = (new_normalized_data_1) / (
#                     np.max(new_normalized_data_1) - np.min(new_normalized_data_1))
#         result = new_normalized_data_1
#
#     number_of_samples = 5
#     flattened_result = result[:number_of_samples].flatten()
#     x_axis = np.linspace(0, 10, len(flattened_result))
#     axes[index].plot(x_axis, flattened_result)
#     axes[index].set_title(legend[index])
#
#     # Customize the appearance of the grid lines
#
#
#     # Set x-axis ticks and labels
#     if index == len(axes) - 1:  # For the last subplot
#         axes[index].set_xticks(np.arange(0, 11, 1))
#         axes[index].set_xlabel('Time(s)')
#         axes[index].grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5)
#     else:  # For other subplots
#         axes[index].set_xticks(np.arange(0, 11, 1))
#         axes[index].grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5)
#
# # Set common labels for all subplots
# #fig.text(0.5, 0.05, 'Time(s)', ha='center', va='center')
# fig.text(0.05, 0.5, 'Normalized Signal amplitude(unitless)', ha='center', va='center', rotation='vertical')
#
# plt.show()



# data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
# data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")
# noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

data_clean_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_normalized.npy")
data_noisy_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_normalized.npy")



# Step 1: Split into training and test sets
#noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

noisy_train_eog, noisy_test_eog, clean_train_eog, clean_test_eog = train_test_split(data_noisy_eog, data_clean_eog, test_size=0.2, random_state=42)




# paths_emg = [
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_three_layer_emg.npy",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_true_5_model_emg.npy",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\cnn_emg_result.npy",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_gru_emg.npy",
#          r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_lstm_emg.npy"]

paths_eog = [
         r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_three_layer_eog.npy",
         r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_true_5_model_eog.npy",
         r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\cnn_eog_result.npy",
         r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_gru_eog.npy",
         r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_lstm_eog.npy"]


flattened_clean = clean_test_eog[:5].flatten()
flattened_noisy = noisy_test_eog[:5].flatten()

fig, axes = plt.subplots(7, 1, sharey='col')
plt.subplots_adjust(hspace=0.75)
legend = ["Clean Data", "Noisy Data - EOG Noise", "AE - 3 Layers", "AE - 5 Layers", "CNN Net", "AE GRU", "LSTM Net"]
x_axis = np.linspace(0, 10, len(flattened_clean))

axes[0].plot(x_axis, flattened_clean)
axes[0].set_title(legend[0])
axes[0].set_xticks(np.arange(0, 11, 1))
axes[0].grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5)
axes[0].set_ylim(-0.05, 0.05)

axes[1].plot(x_axis, flattened_noisy)
axes[1].set_title(legend[1])
axes[1].set_xticks(np.arange(0, 11, 1))
axes[1].grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5)
axes[1].set_ylim(-0.05, 0.05)

for index, path in enumerate(paths_eog):
    index += 2
    result = np.load(path)

    if(result.ndim == 3):
        result = result.squeeze(-1)

    number_of_samples = 5
    flattened_result = result[:number_of_samples].flatten()
    axes[index].plot(x_axis, flattened_result)
    axes[index].set_title(legend[index])
    axes[index].set_ylim(-0.05, 0.05)

    # Customize the appearance of the grid lines


    # Set x-axis ticks and labels
    if index == len(axes) - 1:  # For the last subplot
        axes[index].set_xticks(np.arange(0, 11, 1))
        axes[index].set_xlabel('Time(s)')
        axes[index].grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5)
    else:  # For other subplots
        axes[index].set_xticks(np.arange(0, 11, 1))
        axes[index].grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5)

# Set common labels for all subplots
#fig.text(0.5, 0.05, 'Time(s)', ha='center', va='center')
fig.text(0.05, 0.5, 'Normalized Signal amplitude(unitless)', ha='center', va='center', rotation='vertical')

plt.show()

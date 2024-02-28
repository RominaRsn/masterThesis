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


def dB_to_linear(dB):
    return 10**(dB / 10)

# data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
# data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")

data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")

noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_GRU_checkPoint.h5")
#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_LSTM_checkPoint.h5")

model_old = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")

result = model.predict(noisy_test[0:1000])
result_old = model_old.predict(noisy_test[0:1000])

for i in range(1,30):
    fig, axes = plt.subplots(nrows=4, ncols=1, sharey='col')

    row_index = i

    axes[0].plot(clean_test[row_index, :], label = 'Clean Data')
    axes[0].set_title('Clean data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')

    #print(smaller_reshaped_data_clean_test[row_index, :].shape)


    axes[1].plot(noisy_test[row_index, :], label = 'Noisy Data with EMG')
    axes[1].set_title('Noisy data with EMG')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')

    #result = model.predict(result)
    #result = result.transpose()

    axes[2].plot(result[row_index, :], label='predicted data with EMG')
    axes[2].set_title('predicted data')
    axes[2].set_ylabel('Signal amplitude')
    axes[2].set_xlabel('Time')

    axes[3].plot(result_old[row_index, :], label='predicted data with EMG')
    axes[3].set_title('predicted data')
    axes[3].set_ylabel('Signal amplitude')
    axes[3].set_xlabel('Time')



    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of clean and noisy data')

    # Adjust layout to prevent overlap
    #plt.tight_layout()

    # Show the plot
    plt.show()
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch
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

#model = model.simpleModel_modified2()
#model.load_weights(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model_modified_simple.h5")



# data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
# data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")


data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")

data_clean_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_cheby_normalized_smaller.npy")
data_noisy_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_cheby_normalized_smaller.npy")



# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

noisy_train_eog, noisy_test_eog, clean_train_eog, clean_test_eog = train_test_split(data_noisy_eog, data_clean_eog, test_size=0.2, random_state=42)

model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
model_eog = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_checkPoint.h5")

result = model.predict(noisy_test[0: 100])

result_eog = model_eog.predict(noisy_test_eog[0: 100])


signalIndexVector = [0, 1, 3, 7, 11, 13, 14, 16, 17]
signalIndexVector = [0, 1, 3, 4]

for i in range(1,30):
    fig, axes = plt.subplots(nrows=5, ncols=1, sharey='col')

    row_index = i

    axes[0].plot(clean_test[row_index, :], label = 'Clean Data')
    axes[0].plot(clean_test_eog[row_index, :], label = 'Clean Data EOG')
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

    #
    axes[3].plot(noisy_test_eog[row_index, :], label ='Noisy data with EOG')
    axes[3].set_title('Noisy data with EOG')
    axes[3].set_ylabel('Signal amplitude')
    axes[3].set_xlabel('Time')

    axes[4].plot(result_eog[row_index, :], label ='predicted data with EOG')
    axes[4].set_title('predicted data with EOG')
    axes[4].set_ylabel('Signal amplitude')
    axes[4].set_xlabel('Time')



    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of clean and noisy data')

    # Adjust layout to prevent overlap
    #plt.tight_layout()

    # Show the plot
    plt.show()

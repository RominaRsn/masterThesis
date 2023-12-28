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



data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")



# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

# Step 2: Split the training set into training and validation sets
#noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_train, clean_train, test_size=0.1, random_state=42)

num_zeros = (0, 12)

# Pad the array with zeros
# padded_noisy_train = np.pad(noisy_train, ((0, 0), num_zeros), mode='constant')
# padded_clean_train = np.pad(clean_train, ((0, 0), num_zeros), mode='constant')

padded_noisy_test = np.pad(noisy_test, ((0, 0), num_zeros), mode='constant')
padded_clean_test = np.pad(clean_test, ((0, 0), num_zeros), mode='constant')

model = load_model( r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint.h5')
model_old = load_model( r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_more_filters.h5')

results = model.predict(padded_noisy_test[0:1000, :])
results_old = model_old.predict(noisy_test[0:1000, :])
results_old = results_old.squeeze(-1)
np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\results_skip_connection.npy", results)

padded_results = np.pad(results_old, ((0, 0), (0, 12)), mode='constant')

signalIndexVector = [0, 1, 3, 7, 11, 13, 14, 16, 17]

for i in signalIndexVector:
    fig, axes = plt.subplots(nrows=4, ncols=1, sharey='col')

    row_index = i
    #row_index = np.random.randint(0, a)
    #col_index = np.random.randint(0, 11520000/500)

    axes[0].plot(padded_clean_test[row_index, :], label = 'Clean Data')
    axes[0].set_title('Clean data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')

    #print(smaller_reshaped_data_clean_test[row_index, :].shape)


    axes[1].plot(padded_noisy_test[row_index, :], label = 'Noisy Data')
    axes[1].set_title('Noisy data')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')

    #result = model.predict(result)
    #result = result.transpose()

    axes[2].plot(results[row_index, :], label='predicted data- skip connection')
    axes[2].set_title('predicted data')
    axes[2].set_ylabel('Signal amplitude')
    axes[2].set_xlabel('Time')

    axes[3].plot(results_old[row_index, :], label='predicted data- classic AE')
    axes[3].set_title('predicted data')
    axes[3].set_ylabel('Signal amplitude')
    axes[3].set_xlabel('Time')



    # axes[4].plot(results[row_index, :], label='predicted data')
    # axes[4].set_title('predicted data')
    # axes[4].set_ylabel('Signal amplitude')
    # axes[4].set_xlabel('Time')


    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of clean and noisy data')

    # Adjust layout to prevent overlap
    #plt.tight_layout()

    # Show the plot
    plt.show()




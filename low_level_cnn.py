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
from scipy.ndimage import convolve1d
from scipy.signal import butter,filtfilt,iirnotch, convolve, cheby2, sosfiltfilt


#model = model.simpleModel_modified2()
#model.load_weights(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model_modified_simple.h5")



# data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
# data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")


data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")



# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

# Step 2: Split the training set into training and validation sets
#noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_train, clean_train, test_size=0.1, random_state=42)

num_zeros = (0, 12)

# Pad the array with zeros
padded_noisy_train = np.pad(noisy_train, ((0, 0), num_zeros), mode='constant')
padded_clean_train = np.pad(clean_train, ((0, 0), num_zeros), mode='constant')

padded_noisy_test = np.pad(noisy_test, ((0, 0), num_zeros), mode='constant')
padded_clean_test = np.pad(clean_test, ((0, 0), num_zeros), mode='constant')


# model_1 = model.low_cnn()
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#
# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\low_level_cnn_checkpoint.h5'
#
# checkpoint = ModelCheckpoint(checkpoint_path,
#                              monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
#                              save_best_only=True,  # Save only if the validation performance improves
#                              mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
#                              verbose=1)
#
# model_1.optimizer.learning_rate = 1e-6
# model_1.fit(
#     noisy_train,
#     clean_train,
#     epochs=2,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[callback, checkpoint],
#     shuffle=True
#
# )
# model_1.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\low_level_cnn_checkpoint.h5")



######################uncomment for retraining the ae model#######################
# num_zeros = (0, 12)
#
# padded_noisy_train = np.pad(noisy_train, ((0, 0), num_zeros), mode='constant')
# padded_clean_train = np.pad(clean_train, ((0, 0), num_zeros), mode='constant')
#
# padded_noisy_test = np.pad(noisy_test, ((0, 0), num_zeros), mode='constant')
# padded_clean_test = np.pad(clean_test, ((0, 0), num_zeros), mode='constant')
#
#
#
# model = model.load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint.h5")
# result_train = model.predict(padded_noisy_train)
#
# sharpening_kernel = np.array([0, -1, 2, -1, 0])
# def sharpenSignal(data):
#     data = np.array(data)
#     data_sharpened = np.empty_like(data)
#     for i in range(0, len(data)):
#         data_sharpened[i, :] = convolve1d(data[i, :], weights=sharpening_kernel, mode='constant', cval=0.0)
#     return data_sharpened
#
# result_train_sharpened = sharpenSignal(result_train)
#
# model.optimizer.learning_rate = 1e-6
# model.fit(
#     result_train_sharpened,
#     padded_clean_train,
#     epochs=1,
#     batch_size=32,
#     validation_split=0.1,
#     shuffle=True
#
# )
# model.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_model\retrain_skip_all.h5")

######################uncomment for retraining the ae model#######################
def resample(data, sampling_rate):
    data = np.array(data)
    resampled_data = np.empty((len(data), sampling_rate))
    for i in range(0, len(data)):
        resampled_data[i, :] = spicy.signal.resample(data[i, :], sampling_rate)
    return resampled_data



model_ae = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint.h5")

result_ae = model_ae.predict(padded_noisy_test[0:1000])
result_ae = result_ae.squeeze(-1)
result_ae = resample(result_ae, 500)

model_classic = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_more_filters.h5")
result_classic = model_classic.predict(noisy_test[0:1000])

model_2s = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint_2s.h5")
result_2s = model_2s.predict(padded_noisy_test[0:1000])
result_2s = result_2s.squeeze(-1)

model_oisk = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint_oisk.h5")
result_oisk = model_oisk.predict(padded_noisy_test[0:1000])
result_oisk = result_oisk.squeeze(-1)

#plotting the results
signalIndexVector = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
#signalIndexVector = [0, 1, 3, 4]

for i in signalIndexVector:
    fig, axes = plt.subplots(nrows=7, ncols=1, sharey='col')

    row_index = i
    #row_index = np.random.randint(0, a)
    #col_index = np.random.randint(0, 11520000/500)

    axes[0].plot(clean_test[row_index, :], label = 'Clean Data')
    axes[0].set_title('Clean data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')

    #print(smaller_reshaped_data_clean_test[row_index, :].shape)


    axes[1].plot(noisy_test[row_index, :], label = 'Noisy Data')
    axes[1].set_title('Noisy data')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')

    #result = model.predict(result)
    #result = result.transpose()

    # axes[2].plot(np.convolve(result_ae[row_index, :], result_classic[row_index, :], mode='full'), label='predicted data- all skip connections')
    # #axes[2].plot(sharpedened_result_1[row_index, :], label='sharpened')
    # axes[2].set_title('predicted data - all skip connections')
    # axes[2].set_ylabel('Signal amplitude')
    # axes[2].set_xlabel('Time')
    # axes[2].legend(loc='lower right')

    #
    axes[3].plot(result_ae[row_index, :], label ='predicted data- only input skip connection')
    #axes[3].plot(sharpedened_result_2[row_index, :], label='sharpened')
    axes[3].set_title('predicted data - only input skip connection')
    axes[3].set_ylabel('Signal amplitude')
    axes[3].set_xlabel('Time')
    axes[3].legend(loc='lower right')

    axes[4].plot(result_classic[row_index, :], label ='predicted data- only input skip connection')
    axes[4].set_title('predicted data - only input skip connection')
    axes[4].set_ylabel('Signal amplitude')
    axes[4].set_xlabel('Time')
    axes[4].legend(loc='lower right')

    axes[5].plot(result_2s[row_index, :], label ='predicted data- only input skip connection')
    axes[5].set_title('predicted data - only input skip connection')
    axes[5].set_ylabel('Signal amplitude')
    axes[5].set_xlabel('Time')
    axes[5].legend(loc='lower right')

    axes[6].plot(result_oisk[row_index, :], label ='predicted data- only input skip connection')
    axes[6].set_title('predicted data - only input skip connection')
    axes[6].set_ylabel('Signal amplitude')
    axes[6].set_xlabel('Time')
    axes[6].legend(loc='lower right')







    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of clean and noisy data')

    # Adjust layout to prevent overlap
    #plt.tight_layout()

    # Show the plot
    plt.show()






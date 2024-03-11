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
from keras.layers import Input, LSTM, Activation, TimeDistributed, Dense


def LSTM_net():
    datanum = 500
    model = tf.keras.Sequential()

    # Input layer
    model.add(Input(shape=(datanum, 1)))

    # LSTM layers
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(80, return_sequences=True))
    model.add(Activation('tanh'))

    # Output layer
    model.add(TimeDistributed(Dense(units=1)))

    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Create the LSTM network
lstm_model = LSTM_net()




def GRU_net():
    datanum = 500
    model = tf.keras.Sequential()
    model.add(Input(shape=(datanum, 1)))
    model.add(layers.GRU(35, return_sequences=True))
    model.add(layers.Activation('tanh'))
    model.add(layers.GRU(29, return_sequences=True))
    model.add(layers.Activation('tanh'))
    model.add(layers.GRU(35, return_sequences=True))
    model.add(layers.Activation('tanh'))

    model.add(layers.TimeDistributed(layers.Dense(units=1)))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


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
    chunk_size = 900
    num_chunks = len(data) // chunk_size

    filtered_data = np.empty_like(data)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = data[start_idx:end_idx]

        # Apply the filter to the chunk
        filtered_chunk = sosfiltfilt(sos, chunk)

        # Store the filtered chunk in the result array
        filtered_data[start_idx:end_idx] = filtered_chunk

    return filtered_data


data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")


# data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
# data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")


# data_clean_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_cheby_normalized_smaller.npy")
# data_noisy_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_cheby_normalized_smaller.npy")
#noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_eog, data_clean_eog, test_size=0.2, random_state=42)



noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

# noisy_train_cheby = cheby2_lowpass(noisy_train)
# noisy_test_cheby = cheby2_lowpass(noisy_test)
# clean_train_cheby = cheby2_lowpass(clean_train)
# clean_test_cheby = cheby2_lowpass(clean_test)


#
#
# smaller_noisy_train = noisy_train[0:1000]
# smaller_clean_train = clean_train[0:1000]


model = GRU_net()
#
#model_gru = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_model.h5")
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\gru_model_EMG.keras'
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
                             save_best_only=True,  # Save only if the validation performance improves
                             mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
                             verbose=1)
model.optimizer.learning_rate = 1e-3
model.fit(
    noisy_train,
    clean_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    callbacks=[checkpoint, callback],
    shuffle=True
)


#model = LSTM_net()

# model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_model.h5")
# #model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\lstm_encoder_bigger.h5")
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_GRU_checkPoint.h5'
# checkpoint = ModelCheckpoint(checkpoint_path,
#                              monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
#                              save_best_only=True,  # Save only if the validation performance improves
#                              mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
#                              verbose=1)
# model.optimizer.learning_rate = 1e-3
# model.fit(
#     noisy_train,
#     clean_train,
#     epochs=2,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[checkpoint, callback],
#     shuffle=True
# )

# model =load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\lstm_encoder.h5")
#
#result = model.predict(noisy_test[0:100])
#
# model_gru = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_checkpoint.h5")
# # # model.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_model.h5")
#result_gru = model_gru.predict(noisy_test[0:100])
#np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_result.npy", result_gru)    # save
# model_encoder = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_more_filters.h5")
# result_encoder = model_encoder.predict(noisy_test[0:100])
# model_cnn = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN.h5")
# result_cnn = model_cnn.predict(noisy_test[0:100])
# model_cnn_bigger_filter = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN_bigger_kernel.h5")
# result_cnn_bigger_filter = model_cnn_bigger_filter.predict(noisy_test[0:100])
# model_cnn_smaller_filter = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\checkPoint_cnn.h5")
# result_cnn_smaller_filter = model_cnn_smaller_filter.predict(noisy_test[0:100])
#
# added_result = result_cnn_smaller_filter + result_encoder
# plt.subplot(3, 1, 1)
# plt.plot(result[10, :, :], label='result')
# plt.plot(clean_train[10, :], label='clean')
# plt.plot(noisy_train[10, :], label='noisy')
# plt.legend()
# plt.title('Comparison at Index 10')
#
# # Plot for index 100
# plt.subplot(3, 1, 2)
# plt.plot(result[50, :, :], label='result')
# plt.plot(clean_test[50, :], label='clean')
# plt.plot(noisy_test[50, :], label='noisy')
# plt.legend()
# plt.title('Comparison at Index 100')
#
# # Plot for index 100 again
# plt.subplot(3, 1, 3)
# plt.plot(result[70, :, :], label='result')
# plt.plot(clean_test[70, :], label='clean')
# plt.plot(noisy_test[70, :], label='noisy')
# plt.legend()
# plt.title('Comparison at Index 100 (again)')
#
# plt.tight_layout()  # Adjust layout for better spacing
# plt.show()


# #sharpening
# sharpening_kernel = np.array([0, -1, 2, -1, 0])
# # #sharpening_kernel = np.array([0, -2, 3, -2, 0]) #intensified, not very good not very bad
# # # Apply the kernel to the signal
# # sharpened_signal = convolve1d(result[0, :], weights=sharpening_kernel, mode='constant', cval=0.0)
#
# signalIndexVector = [0, 1, 3, 4, 5, 6, 7, 8, 54, 35, 67, 71, 23, 12, 88, 93, 44, 22]
# #
# for i in signalIndexVector:
#     fig, axes = plt.subplots(nrows=4, ncols=1, sharey='col')
#
#     row_index = i
#     #row_index = np.random.randint(0, a)
#     #col_index = np.random.randint(0, 11520000/500)
#
#     axes[0].plot(clean_test[row_index, :], label = 'Clean Data')
#     axes[0].set_title('Clean data')
#     axes[0].set_ylabel('Signal amplitude')
#     axes[0].set_xlabel('Time')
#
#     #print(smaller_reshaped_data_clean_test[row_index, :].shape)
#
#
#     axes[1].plot(noisy_test[row_index, :], label = 'Noisy Data')
#     axes[1].set_title('Noisy data')
#     axes[1].set_ylabel('Signal amplitude')
#     axes[1].set_xlabel('Time')
#
#     #result = model.predict(result)
#     #result = result.transpose()
#
#     axes[2].plot(result[row_index, :], label='predicted data with gru')
#     #axes[2].plot(convolve1d(result_gru[row_index, :], weights=sharpening_kernel, mode='constant', cval=0.0), label='sharpened predicted data with CNN- big filter')
#     axes[2].set_title('predicted data with gru')
#     axes[2].set_ylabel('Signal amplitude')
#     axes[2].set_xlabel('Time')
#     plt.legend()
#
#     axes[3].plot(result_gru[row_index, :], label='predicted data with encoder')
#     axes[3].set_title('predicted data with encoder')
#     axes[3].set_ylabel('Signal amplitude')
#     axes[3].set_xlabel('Time')
# #
# #     axes[4].plot(result_cnn_smaller_filter[row_index, :], label='predicted data with CNN- small filter')
# #     axes[4].set_title('predicted data with CNN- small filter')
# #     axes[4].set_ylabel('Signal amplitude')
# #     axes[4].set_xlabel('Time')
# #
# #     axes[5].plot(result_cnn_bigger_filter[row_index, :], label='predicted data with CNN- bigger filter')
# #     axes[5].set_title('predicted data with CNN- bigger filter')
# #     axes[5].set_ylabel('Signal amplitude')
# #     axes[5].set_xlabel('Time')
# #
# #     axes[6].plot(result_cnn[row_index, :], label='predicted data with CNN')
# #     axes[6].set_title('predicted data with CNN')
# #     axes[6].set_ylabel('Signal amplitude')
# #     axes[6].set_xlabel('Time')
# #
# #     axes[7].plot(np.add(np.ravel(result_encoder[row_index, :]), np.ravel(result_cnn_smaller_filter[row_index, :])), label='ae & cnn small kernel')
# #     axes[7].set_title('ae &cnn small kernel')
# #     axes[7].set_ylabel('Signal amplitude')
# #     axes[7].set_xlabel('Time')
# #
# #
# #     axes[8].plot(np.add(np.ravel(result_gru[row_index, :]), np.ravel(result_cnn_smaller_filter[row_index, :])), label='gru & CNN small kernel')
# #     axes[8].set_title('predicted data with GRU')
# #     axes[8].set_ylabel('Signal amplitude')
# #     axes[8].set_xlabel('Time')
# #
# #
# #
# #
# #     plt.legend()
#     plt.show()

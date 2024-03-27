from keras.constraints import max_norm
from keras.layers import LeakyReLU, Add
from keras import layers,Sequential
from keras.models import *
from keras.layers import *
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
import numpy as np


# def model():
#     # Define the input layer
#     input_layer = Input(shape=(500, 1))  # Assuming 1 channel (e.g., for time series data)
#
#     # Encoding layers
#     encoded1 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(input_layer)
#     encoded1 = MaxPooling1D(2, padding='same')(encoded1)
#
#     encoded2 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded1)
#     encoded2 = MaxPooling1D(2, padding='same')(encoded2)
#
#     encoded3 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded2)
#     encoded3 = MaxPooling1D(2, padding='same')(encoded3)
#
#     encoded4 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded3)
#     encoded4 = MaxPooling1D(2, padding='same')(encoded4)
#
#     encoded5 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded4)
#     encoded5 = MaxPooling1D(2, padding='same')(encoded5)
#
#     encoded6 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded5)
#
#     # Decoding layers (symmetric to the encoding layers)
#     decoded6 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(encoded6)
#     decoded6 = UpSampling1D(2)(decoded6)
#
#     decoded5 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(decoded6)
#     decoded5 = UpSampling1D(2)(decoded5)
#
#     decoded4 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='same')(decoded5)
#     decoded4 = UpSampling1D(2)(decoded4)
#
#     decoded3= Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='valid')(decoded4)
#     decoded3 = UpSampling1D(2)(decoded3)
#
#     decoded2 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='valid')(decoded3)
#     decoded2 = UpSampling1D(2)(decoded2)
#     #
#     # decoded1 = Conv1D(256, 3, activation='relu',kernel_initializer='he_uniform', padding='valid')(decoded2)
#     # decoded1 = UpSampling1D(2)(decoded1)
#
#     output_layer = Conv1D(1, 3, activation='tanh',kernel_initializer='he_uniform', padding='same')(decoded2)  # 1 channel for reconstruction
#
#     # Create the autoencoder model
#     autoencoder = Model(input_layer, output_layer)
#
#     # Compile the autoencoder
#     autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#
#     # Print the summary of the autoencoder model
#     autoencoder.summary()
#     return autoencoder
#
# model = model()
#
# data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
# data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")
#
# noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)
#
#
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#
# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\model_with_5_layers_true_arch_EMG.h5'
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
#     epochs=2,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[callback, checkpoint],
#     shuffle=True
#
# )
#
# del data_clean_normalized, data_noisy_normalized, noisy_train, noisy_test, clean_train, clean_test  # Free up memory
#
data_clean_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_normalized.npy")
data_noisy_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_normalized.npy")



# Step 1: Split into training and test sets
#noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

noisy_train_eog, noisy_test_eog, clean_train_eog, clean_test_eog = train_test_split(data_noisy_eog, data_clean_eog, test_size=0.2, random_state=42)

model = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\model_with_5_layers_true_arch_EMG.h5')

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)

checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\model_with_5_layers_true_arch_EMG_EOG.h5'

checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
                             save_best_only=True,  # Save only if the validation performance improves
                             mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
                             verbose=1)

model.optimizer.learning_rate = 1e-6
model.fit(
    noisy_train_eog,
    clean_train_eog,
    epochs=2,
    batch_size=32,
    validation_split=0.1,
    callbacks=[callback, checkpoint],
    shuffle=True
)

result_eog = model.predict(noisy_test_eog)
np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_true_5_model_eog.npy', result_eog)

del data_clean_eog, data_noisy_eog, noisy_train_eog, noisy_test_eog, clean_train_eog, clean_test_eog  # Free up memory

data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")

noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

result_emg = model.predict(noisy_test)
np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_true_5_model_emg.npy', result_emg)

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch
from keras.utils import plot_model
import os
import masterThesis.model as model
import spicy
import tensorflow as tf
import pickle
import masterThesis.metrics as metrics
import neurokit2 as nk
from keras.callbacks import ModelCheckpoint


#model = model.paper_cnn()
model = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\paper_cnn_checkpoint.h5')

data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")



# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)


# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\paper_cnn_checkpoint.h5'
#
# checkpoint = ModelCheckpoint(checkpoint_path,
#                              monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
#                              save_best_only=True,  # Save only if the validation performance improves
#                              mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
#                              verbose=1)
#
# model.optimizer.learning_rate = 1e-6
#
# model.fit(
#     noisy_train,
#     clean_train,
#     epochs=3,
#     batch_size=16,
#     validation_split=0.1,
#     callbacks=[checkpoint],
#     shuffle=True
# )



data_clean_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_cheby_normalized_smaller.npy")
data_noisy_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_cheby_normalized_smaller.npy")

noisy_train_eog, noisy_test_eog, clean_train_eog, clean_test_eog = train_test_split(data_noisy_eog, data_clean_eog, test_size=0.2, random_state=42)

model = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\paper_cnn_checkpoint.h5')

checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\paper_CNN_retrainWithEOG_LSTM_checkPoint.h5'

checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
                             save_best_only=True,  # Save only if the validation performance improves
                             mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
                             verbose=1)

model.optimizer.learning_rate = 1e-6
model.fit(
    noisy_train_eog,
    clean_train_eog,
    epochs=5,
    batch_size=16,
    validation_split=0.1,
    callbacks=[checkpoint],
    shuffle=True

)


# plt.plot(noisy_train[0])
# plt.plot(clean_train[0])
# plt.plot(model.predict(noisy_train[0:16])[0])
# plt.show()
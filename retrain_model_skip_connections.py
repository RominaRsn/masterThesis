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


num_zeros = (0, 12)

padded_noisy_train = np.pad(noisy_train, ((0, 0), num_zeros), mode='constant')
padded_clean_train = np.pad(clean_train, ((0, 0), num_zeros), mode='constant')

padded_noisy_test = np.pad(noisy_test, ((0, 0), num_zeros), mode='constant')
padded_clean_test = np.pad(clean_test, ((0, 0), num_zeros), mode='constant')



model = model.load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\ae_skip_layers_checkpoint.h5")
result_train = model.predict(noisy_train)

sharpening_kernel = np.array([0, -1, 2, -1, 0])
def sharpenSignal(data):
    data = np.array(data)
    data_sharpened = np.empty_like(data)
    for i in range(0, len(data)):
        data_sharpened[i, :] = convolve1d(data[i, :], weights=sharpening_kernel, mode='constant', cval=0.0)
    return data_sharpened

result_train_sharpened = sharpenSignal(result_train)

model.optimizer.learning_rate = 1e-6
model.fit(
    result_train_sharpened,
    padded_clean_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    shuffle=True

)
model.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_model\retrain_skip_all.h5")


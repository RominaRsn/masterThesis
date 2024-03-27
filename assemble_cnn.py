import matplotlib.pyplot as plt
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



# result = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\cnn_emg_result.npy")
result_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\cnn_eog_result.npy")

# plt.plot(result[0,:])
# plt.show()


data_clean_normalized_cheby = np.load(
    r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_normalized.npy")
data_noisy_normalized_cheby = np.load(
    r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_normalized.npy")

noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby,
                                                                    data_clean_normalized_cheby, test_size=0.2,
                                                                    random_state=42)
plt.plot(clean_test[0,:])
plt.plot(noisy_test[0,:])
plt.plot(result_eog[0,:])
plt.legend(['clean', 'noisy', 'denoised'])
plt.show()




from keras.constraints import max_norm
from keras.layers import LeakyReLU, Add
from keras import layers,Sequential
from keras.models import *
from keras.layers import *
import numpy as np
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
import matplotlib.pyplot as plt




data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")

noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

result_emg = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\cnn_emg_result.npy")

plt.plot(result_emg[0])
plt.plot(clean_test[0])
plt.plot(noisy_test[0])
plt.legend(['result_emg', 'clean_test', 'noisy_test'])
plt.show()

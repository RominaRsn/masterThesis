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



model = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\model_with_3_layers_paper_arch_EMG.h5')


data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")

noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

result_emg = model.predict(noisy_test)
np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_three_layer_emg.npy", result_emg)

del data_clean_normalized, data_noisy_normalized, noisy_train, noisy_test, clean_train, clean_test  # Free up memory

#
data_clean_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog_normalized.npy")
data_noisy_eog = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog_normalized.npy")



# Step 1: Split into training and test sets
#noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

noisy_train_eog, noisy_test_eog, clean_train_eog, clean_test_eog = train_test_split(data_noisy_eog, data_clean_eog, test_size=0.2, random_state=42)

result_eog = model.predict(noisy_test_eog)
np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrained_models_no_cheby_filter\result_three_layer_eog.npy", result_eog)

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch, cheby2, sosfiltfilt
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


def dB_to_linear(dB):
    return 10**(dB / 10)

# data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
# data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")

data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")

noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_GRU_checkPoint.h5")
#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_LSTM_checkPoint.h5")

#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\ae_skip_layers_oisk.h5")


padding_needed = (0, 12)  # Add 12 columns of padding to the second dimension to make it 512

# Pad the matrix
padded_matrix = np.pad(noisy_test[0:1000], ((0, 0), (0, 12), (0, 0)), mode='constant', constant_values=0)

result = model.predict(padded_matrix)
result = result.squeeze(-1)

cornoisyclean_list = []
corcleaned_list = []
snrnoisy_list = []
snrcleaned_list = []
snrnoisy_notdB_list = []
snrclean_notdB_list = []
rrmseNoisy_list = []
rrmseCleaned_list = []
rmsNoisy_list = []
rmsCleaned_list = []
Psnr_clean_list = []
Psnr_cleaned_list = []


for i in range(result.shape[0]):
    filtered_instance = result[i, :]
    noisy_instance = noisy_test[i, :]
    clean_instance = clean_test[i, :]

    cornoisyclean = np.corrcoef(clean_instance, noisy_instance)
    cornoisyclean_list.append(cornoisyclean)
    corcleaned = np.corrcoef(clean_instance, filtered_instance)
    corcleaned_list.append(cornoisyclean)


    snrnoisy = metrics.metrics.snr(clean_instance, noisy_instance)
    snrnoisy_list.append(snrnoisy)
    snrcleaned = metrics.metrics.snr(clean_instance, filtered_instance)
    snrcleaned_list.append(snrcleaned)

    snr_nosiy_not_db = dB_to_linear(snrnoisy)
    snrnoisy_notdB_list.append(snr_nosiy_not_db)
    snr_cleaned_not_db = dB_to_linear(snrcleaned)
    snrclean_notdB_list.append(snr_cleaned_not_db)

    rrmseNoisy = metrics.metrics.rrmseMetric(clean_instance, noisy_instance)
    rrmseNoisy_list.append(rrmseNoisy)
    rrmseCleaned = metrics.metrics.rrmseMetric(clean_instance, filtered_instance)
    rrmseCleaned_list.append(rrmseCleaned)

    diffNoisyClean = noisy_instance - clean_instance
    rmsNoisy = np.sqrt(np.mean(diffNoisyClean ** 2))
    rmsNoisy_list.append(rmsNoisy)

    diffCleanedClean = filtered_instance - clean_instance
    rmsCleaned = np.sqrt(np.mean(diffCleanedClean ** 2))
    rmsCleaned_list.append(rmsCleaned)

    if(~np.isnan(max(clean_instance) / rrmseNoisy ** 2) and ~np.isnan(max(clean_instance) / rrmseCleaned ** 2)):
        Psnr_noisy = 10 * np.log10(max(clean_instance) / rrmseNoisy ** 2)
        Psnr_cleaned = 10 * np.log10(max(clean_instance) / rrmseCleaned ** 2)
        Psnr_clean_list.append(Psnr_noisy)

        Psnr_cleaned_list.append(Psnr_cleaned)







# Convert lists to NumPy arrays
cornoisyclean_arr = np.array(cornoisyclean_list)
corcleaned_arr = np.array(corcleaned_list)
snrnoisy_arr = np.array(snrnoisy_list)
snrcleaned_arr = np.array(snrcleaned_list)
snrnoisy_notdB_arr = np.array(snrnoisy_notdB_list)
snrclean_notdB_arr = np.array(snrclean_notdB_list)
rrmseNoisy_arr = np.array(rrmseNoisy_list)
rrmseCleaned_arr = np.array(rrmseCleaned_list)
rmsNoisy_arr = np.array(rmsNoisy_list)
rmsCleaned_arr = np.array(rmsCleaned_list)
Psnr_clean_arr = np.array(Psnr_clean_list)
Psnr_cleaned_arr = np.array(Psnr_cleaned_list)

# Calculate averages
cornoisyclean_avg = np.mean(cornoisyclean_arr)
corcleaned_avg = np.mean(corcleaned_arr)
snrnoisy_avg = np.mean(snrnoisy_arr)
snrcleaned_avg = np.mean(snrcleaned_arr)
snrnoisy_notdB_avg = np.mean(snrnoisy_notdB_arr)
snrclean_notdB_avg = np.mean(snrclean_notdB_arr)
rrmseNoisy_avg = np.mean(rrmseNoisy_arr)
rrmseCleaned_avg = np.mean(rrmseCleaned_arr)
rmsNoisy_avg = np.mean(rmsNoisy_arr)
rmsCleaned_avg = np.mean(rmsCleaned_arr)
Psnr_clean_avg = np.mean(Psnr_clean_arr)
Psnr_cleaned_avg = np.mean(Psnr_cleaned_arr)

# Print averages
print("Average cornoisyclean:", cornoisyclean_avg)
print("Average corcleaned:", corcleaned_avg)
print("Average snrnoisy:", snrnoisy_avg)
print("Average snrcleaned:", snrcleaned_avg)
print("Average snrnoisy_notdB:", snrnoisy_notdB_avg)
print("Average snrclean_notdB:", snrclean_notdB_avg)
print("Average rrmseNoisy:", rrmseNoisy_avg)
print("Average rrmseCleaned:", rrmseCleaned_avg)
print("Average rmsNoisy:", rmsNoisy_avg)
print("Average rmsCleaned:", rmsCleaned_avg)
print("Average Psnr_clean:", Psnr_clean_avg)
print("Average Psnr_cleaned:", Psnr_cleaned_avg)
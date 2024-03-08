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

# data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
# data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")


data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")

noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_GRU_checkPoint.h5")
#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_LSTM_checkPoint.h5")

#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_checkPoint.h5")

model_paths = [
    r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_checkPoint.h5",
    r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_LSTM_checkPoint.h5",
    r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_GRU_checkPoint.h5",
    # Add more model paths as needed
    r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\paper_CNN_retrainWithEOG_LSTM_checkPoint.h5'
]

# Iterate over each model
for model_path in model_paths:
    # Load model
    model = load_model(model_path)

    # Generate predictions
    result = model.predict(noisy_test)


    if(result.ndim == 3):
        result.squeeze(-1)

    np.save(fr"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\nonChebyResults\result_{os.path.basename(model_path)[:-3]}.npy", result)
    # Filter signals
    # filteredSignal_45 = nk.signal_filter(noisy_test, sampling_rate=250, highcut=40, method='butterworth', order=4)
    # filteredSignal_30 = nk.signal_filter(noisy_test, sampling_rate=250, highcut=30, method='butterworth', order=4)
    # filteredSignal_70 = nk.signal_filter(noisy_test, sampling_rate=250, highcut=70, method='butterworth', order=4)

    # Calculate metrics
    clean_input_test_vec = np.ravel(clean_test)
    noisy_input_test_vec = np.ravel(noisy_test)
    test_reconstructions_vec = np.ravel(result)
    cornoisyclean = np.corrcoef(clean_input_test_vec, noisy_input_test_vec)
    corcleaned = np.corrcoef(clean_input_test_vec, test_reconstructions_vec)

    snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
    snrcleaned = metrics.metrics.snr(clean_input_test_vec, test_reconstructions_vec)

    snr_nosiy_not_db = dB_to_linear(snrnoisy)
    snr_cleaned_not_db = dB_to_linear(snrcleaned)

    rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
    rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, test_reconstructions_vec)

    diffNoisyClean = noisy_test - clean_test
    rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))

    Psnr_clean = 10*np.log10(max(clean_input_test_vec)/rrmseNoisy **2)

    #diffCleanedClean = test_reconstructions_vec.reshape(test_reconstructions_vec.shape[0], -1) - clean_test
    diffCleanedClean = test_reconstructions_vec - clean_input_test_vec

    rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))
    Psnr_cleaned = 10*np.log10(max(clean_input_test_vec)/rrmseCleaned **2)

    # Get the user's home directory
    user_home = os.path.expanduser("~")
    # Specify the file path in the Downloads directory
    file_path = os.path.join(user_home, "Downloads", "EMG_With_Noise_All_models_non_cheby_data.txt")

    # Write results to file
    with open(file_path, 'a') as fm:
        fm.write("----------------- Results for Model: {} ---------------------\n".format(model_path))
        fm.write("SNRNoisy: {}\n".format(snrnoisy))
        fm.write("SNRCleaned: {}\n".format(snrcleaned))
        fm.write("RMSNoisy: {}\n".format(rmsNoisy))
        fm.write("RMSCleaned: {}\n".format(rmsCleaned))
        fm.write("RMSENoisy: {}\n".format(rrmseNoisy))
        fm.write("RMSECleaned: {}\n".format(rrmseCleaned))
        fm.write("PearsonCorrNoisy: {}\n".format(cornoisyclean[0, 1]))
        fm.write("PearsonCorrCleaned: {}\n".format(corcleaned[0, 1]))
        fm.write("SNRNoisyNotDB: {}\n".format(snr_nosiy_not_db))
        fm.write("SNRCleanedNotDB: {}\n".format(snr_cleaned_not_db))
        fm.write("PSNRClean: {}\n".format(Psnr_clean))
        fm.write("PSNRFiltered: {}\n".format(Psnr_cleaned))
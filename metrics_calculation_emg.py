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

model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_checkPoint.h5")

result = model.predict(noisy_test)

filteredSignal_45 = nk.signal_filter(noisy_test, sampling_rate=250, highcut=40, method='butterworth', order=4)
filteredSignal_30 = nk.signal_filter(noisy_test, sampling_rate=250, highcut=30, method='butterworth', order=4)
filteredSignal_70 = nk.signal_filter(noisy_test, sampling_rate=250, highcut=70, method='butterworth', order=4)



#Calculating the metrics
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


#compute rmse between noisy and clean test data
diffNoisyClean = noisy_test - clean_test
rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))

#computing Psnr
Psnr_clean = 10*np.log10(max(clean_input_test_vec)/rrmseNoisy **2)

#compute rmse bwtween cleaned and clean test data
test_reconstructions = result.reshape((result.shape[0], result.shape[1]))
diffCleanedClean = test_reconstructions - clean_test
rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))

#computing Psnr
Psnr_cleaned = 10*np.log10(max(clean_input_test_vec)/rrmseCleaned **2)


# Get the user's home directory
user_home = os.path.expanduser("~")
# Specify the file path in the Downloads directory
file_path = os.path.join(user_home, "Downloads", "AE_EMG_With_noise")

fm = open(file_path, 'w')

fm.write("-----------------The results with EMG noise---------------------\n")

fm.write("Filtred signal with AutoEncoder\n")
fm.write("SNRNoisy: %f\n" % snrnoisy);
fm.write("SNRCleaned: %f\n" % snrcleaned);
fm.write("RMSNoisy: %f\n" % rmsNoisy);
fm.write("RMSCleaned: %f\n" % rmsCleaned);
fm.write("RMSENoisy: %f\n" % rrmseNoisy);
fm.write("RMSECleaned: %f\n" % rrmseCleaned);
fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
fm.write("SNRNoisyNotDB: %f\n" % snr_nosiy_not_db);
fm.write("SNRCleanedNotDB: %f\n" % snr_cleaned_not_db);
fm.write("PSNRClean: %f\n" % Psnr_clean);
fm.write("PSNRFiltered: %f\n" % Psnr_cleaned);
fm.close()


#compute rmse between noisy and clean test data
diffNoisyClean = noisy_test-clean_test
rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))

#compute rmse between noisy and clean test data
diffNoisyClean = noisy_test-clean_test
rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))



filtered_Signal = np.ravel(filteredSignal_45)

#compute rmse bwtween cleaned and clean test data
diffCleanedClean = filtered_Signal-clean_input_test_vec
rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))

cornoisyclean = np.corrcoef(clean_input_test_vec , noisy_input_test_vec)
corcleaned = np.corrcoef(clean_input_test_vec , filtered_Signal)
snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
snrcleaned = metrics.metrics.snr(clean_input_test_vec, filtered_Signal)

snr_nosiy_not_db = dB_to_linear(snrnoisy)
snr_cleaned_not_db = dB_to_linear(snrcleaned)
#covnoisyclean= np.cov(noisy_input_test, pure_input_test)
#covcleanedclean = np.cov(reconstruction,pure_input_test)

rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)

#computing Psnr
Psnr_cleaned = 10*np.log10(max(clean_input_test_vec)/rrmseCleaned **2)

#plt.figure()
#plt.plot(filteredsignal)
#plt.figure()
#plt.plot(noisy_input_vec)

fm = open(file_path, 'a')
fm.write("Filtred signal with BW filter 45Hz\n")
fm.write("SNRNoisy: %f\n" % snrnoisy);
fm.write("SNRCleaned: %f\n" % snrcleaned);
fm.write("RMSNoisy: %f\n" % rmsNoisy);
fm.write("RMSCleaned: %f\n" % rmsCleaned);
fm.write("RMSENoisy: %f\n" % rrmseNoisy);
fm.write("RMSECleaned: %f\n" % rrmseCleaned);
fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
fm.write("SNRNoisyNotDB: %f\n" % snr_nosiy_not_db);
fm.write("SNRCleanedNotDB: %f\n" % snr_cleaned_not_db);
fm.write("PSNRClean: %f\n" % Psnr_clean);
fm.write("PSNRFiltered: %f\n" % Psnr_cleaned);




#compute rmse between noisy and clean test data
diffNoisyClean = noisy_test-clean_test
rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))


filtered_Signal = np.ravel(filteredSignal_30)

#compute rmse bwtween cleaned and clean test data
diffCleanedClean = filtered_Signal-clean_input_test_vec
rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))

cornoisyclean = np.corrcoef(clean_input_test_vec , noisy_input_test_vec)
corcleaned = np.corrcoef(clean_input_test_vec , filtered_Signal)
snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
snrcleaned = metrics.metrics.snr(clean_input_test_vec, filtered_Signal)

snr_nosiy_not_db = dB_to_linear(snrnoisy)
snr_cleaned_not_db = dB_to_linear(snrcleaned)
#covnoisyclean= np.cov(noisy_input_test, pure_input_test)
#covcleanedclean = np.cov(reconstruction,pure_input_test)

rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)

#computing Psnr
Psnr_cleaned = 10*np.log10(max(clean_input_test_vec)/rrmseCleaned **2)

#plt.figure()
#plt.plot(filteredsignal)
#plt.figure()
#plt.plot(noisy_input_vec)

fm = open(file_path, 'a')
fm.write("Filtred signal with BW filter 30Hz\n")
fm.write("SNRNoisy: %f\n" % snrnoisy);
fm.write("SNRCleaned: %f\n" % snrcleaned);
fm.write("RMSNoisy: %f\n" % rmsNoisy);
fm.write("RMSCleaned: %f\n" % rmsCleaned);
fm.write("RMSENoisy: %f\n" % rrmseNoisy);
fm.write("RMSECleaned: %f\n" % rrmseCleaned);
fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
fm.write("SNRNoisyNotDB: %f\n" % snr_nosiy_not_db);
fm.write("SNRCleanedNotDB: %f\n" % snr_cleaned_not_db);
fm.write("PSNRClean: %f\n" % Psnr_clean);
fm.write("PSNRFiltered: %f\n" % Psnr_cleaned);






#compute rmse between noisy and clean test data
diffNoisyClean = noisy_test-clean_test
rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))


filtered_Signal = np.ravel(filteredSignal_70)

#compute rmse bwtween cleaned and clean test data
diffCleanedClean = filtered_Signal-clean_input_test_vec
rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))

cornoisyclean = np.corrcoef(clean_input_test_vec , noisy_input_test_vec)
corcleaned = np.corrcoef(clean_input_test_vec , filtered_Signal)
snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
snrcleaned = metrics.metrics.snr(clean_input_test_vec, filtered_Signal)

snr_nosiy_not_db = dB_to_linear(snrnoisy)
snr_cleaned_not_db = dB_to_linear(snrcleaned)
#covnoisyclean= np.cov(noisy_input_test, pure_input_test)
#covcleanedclean = np.cov(reconstruction,pure_input_test)

rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)

#computing Psnr
Psnr_cleaned = 10*np.log10(max(clean_input_test_vec)/rrmseCleaned **2)

#plt.figure()
#plt.plot(filteredsignal)
#plt.figure()
#plt.plot(noisy_input_vec)

fm = open(file_path, 'a')
fm.write("Filtred signal with BW filter 70Hz\n")
fm.write("SNRNoisy: %f\n" % snrnoisy);
fm.write("SNRCleaned: %f\n" % snrcleaned);
fm.write("RMSNoisy: %f\n" % rmsNoisy);
fm.write("RMSCleaned: %f\n" % rmsCleaned);
fm.write("RMSENoisy: %f\n" % rrmseNoisy);
fm.write("RMSECleaned: %f\n" % rrmseCleaned);
fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
fm.write("SNRNoisyNotDB: %f\n" % snr_nosiy_not_db);
fm.write("SNRCleanedNotDB: %f\n" % snr_cleaned_not_db);
fm.write("PSNRClean: %f\n" % Psnr_clean);
fm.write("PSNRFiltered: %f\n" % Psnr_cleaned);


fm.close()


import tensorflow
import keras
from keras.models import Sequential, save_model
from keras.layers import Conv1D, Conv1DTranspose
from keras.constraints import max_norm
from scipy.signal import butter,filtfilt,iirnotch
from keras.utils import plot_model
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
import mne
import asa_export
import EEGDNoiseNetModels

def loadsmallclean():
    EEG_Clean1 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch1.vhdr")
    signals = mne.ioraw_data = EEG_Clean1.get_data()
    sampling_freq = 250
    return (signals, sampling_freq)

def loadsmallnoisy():
    EEG_Noisy= mne.io.read_raw_brainvision(r"C:\Users\RominaRsn\PycharmProjects\masterThesis\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_1.vhdr")
    noisy_signals = mne.ioraw_data = EEG_Noisy.get_data()
    sampling_freq = 250
    return (noisy_signals, sampling_freq)

def loadclean():
    EEG_Clean1 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch1.vhdr")
    EEG_Clean2 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch2.vhdr")
    EEG_Clean3 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch3.vhdr")
    signals = mne.ioraw_data = EEG_Clean1.get_data()

    clean_data2 = mne.ioraw_data = EEG_Clean2.get_data()
    signals = np.append(signals, clean_data2, axis=1)
    clean_data2 = mne.ioraw_data = EEG_Clean3.get_data()
    signals = np.append(signals, clean_data2, axis=1)
    sampling_freq = 250

    return (signals, sampling_freq)

def loadnoisy():
    EEG_NoisyF1 = mne.io.read_raw_brainvision(r"C:\Users\RominaRsn\PycharmProjects\masterThesis\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_1.vhdr")
    EEG_NoisyF2 = mne.io.read_raw_brainvision(r"C:\Users\RominaRsn\PycharmProjects\masterThesis\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_2.vhdr")
    EEG_NoisyF3 = mne.io.read_raw_brainvision(r"C:\Users\RominaRsn\PycharmProjects\masterThesis\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_3.vhdr")

    noisy_signals = mne.ioraw_data = EEG_NoisyF1.get_data()
    noisy_dataF4 = mne.ioraw_data = EEG_NoisyF2.get_data()
    noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)

    noisy_dataF4 = mne.ioraw_data = EEG_NoisyF3.get_data()
    noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)
    sampling_freq = 250
    return (noisy_signals, sampling_freq)

def loadnewnoisy():
    EEG_NoisyF1 = mne.io.read_raw_brainvision('/Users/matthiasdumpelmann/data/EEG/simultan/subclinical182_newnoisescaledfhp30iir_4_1.vhdr')
    EEG_NoisyF2 = mne.io.read_raw_brainvision('/Users/matthiasdumpelmann/data/EEG/simultan/subclinical182_newnoisescaledfhp30iir_4_2.vhdr')
    EEG_NoisyF3 = mne.io.read_raw_brainvision('/Users/matthiasdumpelmann/data/EEG/simultan/subclinical182_newnoisescaledfhp30iir_4_3.vhdr')

    noisy_signals = mne.ioraw_data = EEG_NoisyF1.get_data()
    noisy_dataF4 = mne.ioraw_data = EEG_NoisyF2.get_data()
    noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)

    noisy_dataF4 = mne.ioraw_data = EEG_NoisyF3.get_data()
    noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)
    sampling_freq = 250
    return (noisy_signals, sampling_freq)

def filtering_signals(data, fs, low_freq, high_freq, notch_freq, order):
    low_wn = low_freq / (fs * 0.5)
    high_wn = high_freq / (fs * 0.5)
    b, a = butter(order, low_wn)
    data = filtfilt(b, a, data, axis=0)
  #  b, a = butter(order, high_wn,'high')
  #  data = filtfilt(b, a, data, axis=0)
  #  b, a = iirnotch(notch_freq, 35, fs)
   # data = filtfilt(b, a, data, axis=1)
   # data = filtfilt(b, a, data, axis=0)
    return data


# SNR

def snr(firstSignal,secondSignal):
    snr = 10*np.log10(np.sum(secondSignal**2)/np.sum((firstSignal-secondSignal)**2))
    return snr


def snrDiff(originalSignal, preprocessedSignal, predictedSignal):
    snrInput = snr(originalSignal, preprocessedSignal)
    snrOutput = snr(predictedSignal, preprocessedSignal)
    snrDiff = snrOutput - snrInput

    return snrDiff


# Pearson correlation coefficient

def accMetric(trueDataChannel, predictedDataChannel):
    dataChannel = np.vstack((trueDataChannel.T, predictedDataChannel.T))
    covarianceMatrix = np.cov(dataChannel)
    covariance = covarianceMatrix[0, 1]
    varianceTrue = covarianceMatrix[0, 0]
    variancePred = covarianceMatrix[1, 1]
    acc = covariance / np.sqrt(varianceTrue * variancePred)

    return acc

# Relative RMSE

def rrmseMetric(yTrue, yPred):
    yDiff = yPred - yTrue
    rmsYDiff = np.sqrt(np.mean(yDiff ** 2))
    rmsYTrue = np.sqrt(np.mean(yTrue ** 2))

    return rmsYDiff / rmsYTrue


# RMSE



# exportcleaned
#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='gpu')

devices = tensorflow.config.list_physical_devices()
print(devices)
# hide GPU
tensorflow.config.set_visible_devices([], 'GPU')


# Model configuration
input_shape = (500, 1)
batch_size = 32
#no_epochs = 30
no_epochs = 1


train_test_split = 0.4
validation_split = 0.1
verbosity = 1
max_norm_value = 6.0

#load EEG
clean_EEG = loadclean()
#clean_EEG = loadsmallclean()
clean_data = clean_EEG[0]
sampling_freq = clean_EEG[1]

#noisy_data = loadnoisy()
noisy_data = loadnoisy()
#noisy_data = loadsmallnoisy()

noisy_dataF3 = noisy_data[0]
num_chanClean, num_SamplesClean = clean_data.shape
nChan = num_chanClean
num_chanNoisy, num_SamplesNoisy = noisy_dataF3.shape
print("********************")
print(clean_data.shape)
print(num_chanClean)
print(num_SamplesClean)


# check dimensions of clean and noisy datafiles
if num_chanClean != num_chanNoisy:
    exit()

if num_SamplesClean != num_SamplesNoisy:
    exit()

pmaxn = np.max(noisy_dataF3)
pminn = np.min(noisy_dataF3)

#pmaxn = np.max(clean_data)
#pminn = np.min(clean_data)

#estimate value range for noisy and clean data
pure_probe = np.zeros((1, sampling_freq))
noisy_probe = np.zeros((1, sampling_freq))
nOffset = sampling_freq/2

for s in range(0, sampling_freq-1):
    pure_probe[0, s] = clean_data[0, s + 120]

for s in range(0, sampling_freq-1):
    noisy_probe[0, s] = noisy_dataF3[0, s + 120]

minc = 3 * np.min(pure_probe)
maxc = 3 * np.max(pure_probe)

minn = 3 * np.min(noisy_probe)
maxn = 3 * np.max(noisy_probe)

#determine number of epochs with data
ns = 128 * 3
#ns = 128
# for small dataset ns = 128
#ns=64
nb = 30

nDataEpochs = ns * nb * num_chanClean
#y_val_noisy = data_noisy[:,1]
y_val_noisy = np.zeros((nDataEpochs,  2 * sampling_freq))
noisy_sample = np.zeros((2 * sampling_freq, 1))
#y_val_pure = data_pure[:,1]
y_val_pure = np.zeros((nDataEpochs, 2 * sampling_freq))
pure_sample = np.zeros((2 * sampling_freq, 1))
# Reshape data
y_val_noisy_r = []
y_val_pure_r = []
# for i in range(0, len(y_val_noisy)):
index = 0;
nSeizureStartOffset = 60 * sampling_freq
stddnoisy = 1.e-25
stddpure = 1.e-25
pmaxn = 1.e-25
pmax = 1.e-25
pminn= -100000000000
pmin = -100000000000
index = 0
for i in range(0, 1):
    imod = np.int_(index / num_chanClean);

    for se in range(0, ns):
        ne = num_chanClean
        seizureOffset = (se * 2 * 60 * sampling_freq) + (10 * sampling_freq)
        for e in range(0, ne):


            for b in range(0, nb):
                blockoffSet = b * 2 * sampling_freq;
                for s in range(0, 2 * sampling_freq):
                    #  pure_sample[s] = clean_data[e, s + seizureOffset + blockoffSet]
                    y_val_pure[e + se * ne, s] = clean_data[e, s + seizureOffset + blockoffSet]

                for s in range(0, 2 * sampling_freq):
                    # noisy_sample[s] = noisy_dataF3[e, s + seizureOffset + blockoffSet]
                    y_val_noisy[e + se * ne, s] = noisy_dataF3[e, s + seizureOffset + blockoffSet]

                noisy_sample = y_val_noisy[e + se * ne]
                pure_sample = y_val_pure[e + se * ne]

               # if index == 0:
               #     pmaxn= np.max(noisy_sample)
               #     pminn= np.min(noisy_sample)

                noisy_sample = (noisy_sample - np.min(noisy_sample)) / (np.max(noisy_sample) - np.min(noisy_sample))
                pure_sample = (pure_sample - np.min(pure_sample)) / (np.max(pure_sample) - np.min(pure_sample))
                #noisy_sample = (noisy_sample - minn) / (maxn - minn)
                #pure_sample = (pure_sample - minc) / (maxc - minc)

                y_val_noisy_r.append(noisy_sample)
                y_val_pure_r.append(pure_sample)

                index = index + 1
y_val_noisy_r = np.array(y_val_noisy_r)
y_val_pure_r = np.array(y_val_pure_r)
noisy_input = y_val_noisy_r.reshape((y_val_noisy_r.shape[0], y_val_noisy_r.shape[1], 1))
pure_input = y_val_pure_r.reshape((y_val_pure_r.shape[0], y_val_pure_r.shape[1], 1))
print(y_val_noisy_r.shape)

# Train/test split
percentage_training = math.floor((1 - train_test_split) * len(noisy_input))
noisy_input, noisy_input_test = noisy_input[:percentage_training], noisy_input[percentage_training:]
pure_input, pure_input_test = pure_input[:percentage_training], pure_input[percentage_training:]


#model

#
# for nBatchSize in range(48, 56, 16):
#     for nLatentNeurons in range (96, 108, 16):
#         # Create the model
#         EComplexCNN = EEGDNoiseNetModels.Complex_CNN(2*sampling_freq)
#
#         model = Sequential()
#         model.add(Conv1D(128, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
#                          kernel_initializer='he_uniform', input_shape=input_shape))
#         model.add(Conv1D(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
#                          kernel_initializer='he_uniform'))
#         model.add(Conv1DTranspose(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
#                                   kernel_initializer='he_uniform'))
#         model.add(Conv1DTranspose(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
#                                   kernel_initializer='he_uniform'))
#         model.add(Conv1D(1, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))
#
#         model.summary()
#
#         # Compile and fit data
#         #model.compile(optimizer='adam', loss='binary_crossentropy')
#         #model.compile(optimizer='adam', loss='mean_absolute_error')
#
#         model.compile(optimizer='adam', loss='mean_squared_error')
#
#         #EComplexCNN.compile(optimizer='adam', loss='mean_squared_error')
#         #model.compile(optimizer='adam', loss='kl_divergence')
#         #model.compile(optimizer='adam', loss='huber_loss')
#         #model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
#         #model.compile(optimizer='adam', loss='cosine_similarity')
#
#         model.fit(noisy_input, pure_input,
#                   epochs=no_epochs,
#                   batch_size=nBatchSize,
#                   validation_split=validation_split)
#
#         #EComplexCNN.fit(noisy_input, pure_input,
#         #          epochs=no_epochs,
#         #          batch_size=nBatchSize,
#         #          verbose=2,
#         #          validation_split=validation_split)
#
#         # save model
#         #model.save('/Users/matthiasdumpelmann/data/EEG/simultan/eegdenoisenetcomplexCNN')
#         #EComplexCNN.save('/Users/matthiasdumpelmann/data/EEG/simultan/eegdenoisenetcomplexCNNModelNoiseLevel4_cpu')
#
#         # Generate reconstructions
#         num_reconstructions = 300
#         # samples = noisy_input_test[:num_reconstructions]
#         # orgs = pure_input_test[:num_reconstructions]
#         samples = noisy_input[:num_reconstructions]
#         orgs = pure_input[:num_reconstructions]
#
#         # samples = noisy_input[:num_reconstructions]
#         #reconstructions = EComplexCNN.predict(samples)
#         reconstructions = model.predict(samples)
#
#         asa_export.exportcleaned(num_chanClean, sampling_freq, noisy_dataF3, EComplexCNN)
#
#         #test the model now on the test data set
#         test_samples = noisy_input_test[:len(noisy_input_test)]
#         #test_reconstructions = EComplexCNN.predict(test_samples)
#         test_reconstructions = model.predict(test_samples)
#
#         asa_export.exportcleaned_testData(num_chanClean, sampling_freq, test_reconstructions)
#
#         asa_export.exportpure_testData(nChan, sampling_freq, pure_input_test)
#
#         asa_export.exportnoisy_testData(nChan, sampling_freq, noisy_input_test)
#
#         #compute rmse between noisy and clean test data
#         diffNoisyClean = noisy_input_test-pure_input_test
#         rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))
#
#         #compute rmse bwtween cleaned and clean test data
#         test_reconstructions = test_reconstructions.reshape((test_reconstructions.shape[0], test_reconstructions.shape[1], 1))
#         diffCleanedClean = test_reconstructions-pure_input_test
#         rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))
#
#         #compute covariance
#         pure_input_test_vec = np.ravel(pure_input_test)
#         noisy_input_vec = np.ravel(noisy_input_test)
#         test_reconstructions_vec = np.ravel(test_reconstructions)
#
#         cornoisyclean = np.corrcoef(pure_input_test_vec , noisy_input_vec)
#         corcleaned = np.corrcoef(pure_input_test_vec , test_reconstructions_vec)
#
#         snrnoisy = snr(pure_input_test_vec, noisy_input_vec)
#         snrcleaned = snr(pure_input_test_vec, test_reconstructions_vec)
#         #covnoisyclean= np.cov(noisy_input_test, pure_input_test)
#         #covcleanedclean = np.cov(reconstruction,pure_input_test)
#
#         rrmseNoisy = rrmseMetric(pure_input_test_vec, noisy_input_vec)
#         rrmseCleaned = rrmseMetric(pure_input_test_vec, test_reconstructions_vec)
#
#         filteredsignal = filtering_signals(noisy_input_vec, 250, 45, 0.5, 50, 4)
#
#         # compute rmse between noisy and clean test data
#         diffNoisyClean = noisy_input_test - pure_input_test
#         rmsNoisy = np.sqrt(np.mean(diffNoisyClean ** 2))
#
#         # compute rmse bwtween cleaned and clean test data
#         diffCleanedClean = filteredsignal - pure_input_test_vec
#         rmsCleaned = np.sqrt(np.mean(diffCleanedClean ** 2))
#
#         cornoisyclean = np.corrcoef(pure_input_test_vec, noisy_input_vec)
#         corcleaned = np.corrcoef(pure_input_test_vec, filteredsignal)
#
#         snrnoisy = snr(pure_input_test_vec, noisy_input_vec)
#         snrcleaned = snr(pure_input_test_vec, filteredsignal)
#         # covnoisyclean= np.cov(noisy_input_test, pure_input_test)
#         # covcleanedclean = np.cov(reconstruction,pure_input_test)
#
#         rrmseNoisy = rrmseMetric(pure_input_test_vec, noisy_input_vec)
#         rrmseCleaned = rrmseMetric(pure_input_test_vec, filteredsignal)
#
#         print("rms noisy snr:" + rrmseNoisy)
#         print("rms cleaned snr:" + rrmseCleaned)
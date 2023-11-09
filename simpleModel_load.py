import keras.models
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch

import masterThesis.model
import metrics
from keras.utils import plot_model
import os
import neurokit2 as nk
import scipy

# clean_data = np.load(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\clean_0.npy')
# noisy_data = np.load(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\noisy_0.npy')
#
# print(np.max(clean_data))
# print(np.max(noisy_data))

# file_path_clean = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\clean_0.npy'
# file_path_noisy = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\noisy_0.npy'

# data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
# data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")

# small_data_clean_normalized = data_clean_normalized[0:1000, :]
# small_data_noisy_normalized = data_noisy_normalized[0:1000, :]
# np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\small_clean_normalized_new.npy', small_data_clean_normalized)
# np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\small_noisy_normalized_new.npy', small_data_noisy_normalized)

data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\small_noisy_normalized_new.npy")
data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\small_clean_normalized_new.npy")

#

# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

# Step 2: Split the training set into training and validation sets
#noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_train, clean_train, test_size=0.1, random_state=42)

# Now, you have:
# - X_train, y_train: Training set
# - X_val, y_val: Validation set
# - X_test, y_test: Test set

# You can print the shapes to check the sizes of the sets
# print("Training set shape:", noisy_train.shape)
# print("Validation set shape:", noisy_val.shape)
# print("Test set shape:", noisy_test.shape)

# reshaped_data_noisy = nosiy_train_t.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)
# reshaped_data_clean = clean_train_t.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)

reshaped_noisy_train = noisy_train.reshape(noisy_train.shape[0], noisy_train.shape[1], 1)
reshaped_clean_train = clean_train.reshape(clean_train.shape[0], clean_train.shape[1], 1)

reshaped_noisy_test = noisy_test.reshape(noisy_test.shape[0], noisy_test.shape[1], 1)
reshaped_clean_test = clean_test.reshape(clean_test.shape[0], clean_test.shape[1], 1)
#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model_simple.h5")
#model = keras.models.load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model_modified_simple_2.h5")
# # Plot the model and save it to a file (optional)
# plot_model(model, to_file='model_plot_simpleModel.png', show_shapes=True, show_layer_names=True)
# model = masterThesis.model.simpleModel()
# callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1)
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.optimizer.learning_rate = 1e-6
#
# model.fit(
#     reshaped_noisy_train,
#     reshaped_clean_train,
#     epochs=5,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[callback],
#     shuffle=True)
#
# model.save('my_model_modified_simple_new_normalization.h5')

#model = keras.models.load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model_modified_simple_new_normalization.h5")

# result = model.predict(reshaped_noisy_test);
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\result.npy", result)

result = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\result.npy")
#
# result = result.squeeze(axis=-1)

# def filter_signal(signal, lowcut, highcut, fs, order=4):
#     sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
#     filtered = scipy.signal.sosfilt(sos, signal)
#     return filtered
#filteredSignal = metrics.metrics.filtering_signals(noisy_test, 250, 45, 0.5, 50, 4)
filteredSignal_45 = nk.signal_filter(noisy_test , sampling_rate=250, lowcut=0.1, highcut=45, method='butterworth', order=4)
filteredSignal_70 = nk.signal_filter(noisy_test, sampling_rate=250, lowcut=0.1, highcut=70, method='butterworth', order=4)
filteredSignal_30 = nk.signal_filter(noisy_test, sampling_rate=250, lowcut=0.1, highcut=30, method='butterworth', order=4)
# spicy.signal.butter(4, [0.1, 45], fs=250, output='sos')
# filteredSignal_45 = metrics.metrics.filtering_signals(noisy_test, 250, 45, 0.5, 50, 4)
# filteredSignal_70 = metrics.metrics.filtering_signals(noisy_test, 250, 70, 0.5, 50, 4)
# filteredSignal_30 = metrics.metrics.filtering_signals(noisy_test, 250, 30, 0.5, 50, 4)

# filteredSignal_45 = filter_signal(noisy_test[0, :], 0.1, 45, 250, 4)
# filteredSignal_70 = filter_signal(noisy_test[0, :], 0.1, 70, 250, 4)
# filteredSignal_30 = filter_signal(noisy_test[0, :], 0.1, 30, 250, 4)

# plt.plot(filteredSignal_45 + .55)
# plt.plot(filteredSignal_70 + .55)
# plt.plot(filteredSignal_30 + .55)
# plt.plot(noisy_test[0, :])
# plt.legend(['45', '70', '30', 'original'])
# plt.show()



#
# #
for i in range(0, 6):
    fig, axes = plt.subplots(nrows=6, ncols=1, sharey='col')

    row_index = i
    max_range = np.max(noisy_test[row_index, :])
    min_range = np.min(noisy_test[row_index, :])


    #col_index = np.random.randint(0, 11520000/500)

    axes[0].plot(noisy_test[row_index, :], label = 'Noisy Data')
    axes[0].set_title('Noisy data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')
    #plt.ylim(min_range, max_range)


    axes[1].plot(clean_test[row_index, :], label = 'clean Data')
    axes[1].set_title('clean data')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')
    #plt.ylim(min_range, max_range)

    axes[2].plot(result[row_index, :], label = 'clean Data_ predicted')
    axes[2].set_title('cleaned data')
    axes[2].set_ylabel('Signal amplitude')
    axes[2].set_xlabel('Time')
    #plt.ylim(min_range, max_range)


    axes[3].plot(filteredSignal_45[row_index, :] + .55, label ='filtered signal-45')
    axes[3].set_title('filtered signal-45')
    axes[3].set_ylabel('Signal amplitude')
    axes[3].set_xlabel('Time')


    axes[4].plot(filteredSignal_30[row_index, :] + .55, label ='filtered signal-30')
    axes[4].set_title('filtered signal-30')
    axes[4].set_ylabel('Signal amplitude')
    axes[4].set_xlabel('Time')

    axes[5].plot(filteredSignal_70[row_index, :] + .55, label ='filtered signal-70')
    axes[5].set_title('filtered signal-70')
    axes[5].set_ylabel('Signal amplitude')
    axes[5].set_xlabel('Time')

    #plt.ylim(min_range, max_range)

    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of clean and noisy data')

    # Adjust layout to prevent overlap
    #plt.tight_layout()

    # Show the plot
    plt.show()
# #
# # Get the user's home directory
# user_home = os.path.expanduser("~")
#
# # Specify the file path in the Downloads directory
# file_path = os.path.join(user_home, "Downloads", "your_file.txt")
#
#
# #Calculating the snr for prediction and clean data
# #for i in range(0,2039):
# #a = masterThesis.metrics.metrics.snr(result, clean_test)
#
#
# #Calculating the metrics
# clean_input_test_vec = np.ravel(clean_test)
# noisy_input_test_vec = np.ravel(noisy_test)
# test_reconstructions_vec = np.ravel(result)
# cornoisyclean = np.corrcoef(clean_input_test_vec, noisy_input_test_vec)
# corcleaned = np.corrcoef(clean_input_test_vec, test_reconstructions_vec)
#
# snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
#
# snrcleaned = metrics.metrics.snr(clean_input_test_vec, test_reconstructions_vec)
#
# rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
# rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, test_reconstructions_vec)
#
#
# #compute rmse between noisy and clean test data
# diffNoisyClean = noisy_test - clean_test
# rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))
#
# #compute rmse bwtween cleaned and clean test data
# test_reconstructions = result.reshape((result.shape[0], result.shape[1]))
# diffCleanedClean = test_reconstructions - clean_test
# rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))
#
# # Get the user's home directory
# user_home = os.path.expanduser("~")
# # Specify the file path in the Downloads directory
# file_path = os.path.join(user_home, "Downloads", "your_file_1.txt")
#
# fm = open(file_path, 'w')
# fm.write("Filtred signal with AutoEncoder\n")
# fm.write("SNRNoisy: %f\n" % snrnoisy);
# fm.write("SNRCleaned: %f\n" % snrcleaned);
# fm.write("RMSNoisy: %f\n" % rmsNoisy);
# fm.write("RMSCleaned: %f\n" % rmsCleaned);
# fm.write("RMSENoisy: %f\n" % rrmseNoisy);
# fm.write("RMSECleaned: %f\n" % rrmseCleaned);
# fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
# fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
# fm.close()
#
#
# #compute rmse between noisy and clean test data
# diffNoisyClean = noisy_test-clean_test
# rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))
#
# #compute rmse between noisy and clean test data
# diffNoisyClean = noisy_test-clean_test
# rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))
#
#
# filtered_Signal = np.ravel(filteredSignal_45)
#
# #compute rmse bwtween cleaned and clean test data
# diffCleanedClean = filtered_Signal-clean_input_test_vec
# rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))
#
# cornoisyclean = np.corrcoef(clean_input_test_vec , noisy_input_test_vec)
# corcleaned = np.corrcoef(clean_input_test_vec , filtered_Signal)
# snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
# snrcleaned = metrics.metrics.snr(clean_input_test_vec, filtered_Signal)
# #covnoisyclean= np.cov(noisy_input_test, pure_input_test)
# #covcleanedclean = np.cov(reconstruction,pure_input_test)
#
# rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
# rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)
#
# #plt.figure()
# #plt.plot(filteredsignal)
# #plt.figure()
# #plt.plot(noisy_input_vec)
#
# fm = open(file_path, 'a')
# fm.write("Filtred signal with BW filter 45Hz\n")
# fm.write("SNRNoisy: %f\n" % snrnoisy);
# fm.write("SNRCleaned: %f\n" % snrcleaned);
# fm.write("RMSNoisy: %f\n" % rmsNoisy);
# fm.write("RMSCleaned: %f\n" % rmsCleaned);
# fm.write("RMSENoisy: %f\n" % rrmseNoisy);
# fm.write("RMSECleaned: %f\n" % rrmseCleaned);
# fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
# fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
#
#
#
#
# #compute rmse between noisy and clean test data
# diffNoisyClean = noisy_test-clean_test
# rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))
#
#
# filtered_Signal = np.ravel(filteredSignal_30)
#
# #compute rmse bwtween cleaned and clean test data
# diffCleanedClean = filtered_Signal-clean_input_test_vec
# rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))
#
# cornoisyclean = np.corrcoef(clean_input_test_vec , noisy_input_test_vec)
# corcleaned = np.corrcoef(clean_input_test_vec , filtered_Signal)
# snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
# snrcleaned = metrics.metrics.snr(clean_input_test_vec, filtered_Signal)
# #covnoisyclean= np.cov(noisy_input_test, pure_input_test)
# #covcleanedclean = np.cov(reconstruction,pure_input_test)
#
# rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
# rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)
#
# #plt.figure()
# #plt.plot(filteredsignal)
# #plt.figure()
# #plt.plot(noisy_input_vec)
#
# fm = open(file_path, 'a')
# fm.write("Filtred signal with BW filter 30Hz\n")
# fm.write("SNRNoisy: %f\n" % snrnoisy);
# fm.write("SNRCleaned: %f\n" % snrcleaned);
# fm.write("RMSNoisy: %f\n" % rmsNoisy);
# fm.write("RMSCleaned: %f\n" % rmsCleaned);
# fm.write("RMSENoisy: %f\n" % rrmseNoisy);
# fm.write("RMSECleaned: %f\n" % rrmseCleaned);
# fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
# fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
#
#
#
#
#
# #compute rmse between noisy and clean test data
# diffNoisyClean = noisy_test-clean_test
# rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))
#
#
# filtered_Signal = np.ravel(filteredSignal_70)
#
# #compute rmse bwtween cleaned and clean test data
# diffCleanedClean = filtered_Signal-clean_input_test_vec
# rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))
#
# cornoisyclean = np.corrcoef(clean_input_test_vec , noisy_input_test_vec)
# corcleaned = np.corrcoef(clean_input_test_vec , filtered_Signal)
# snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
# snrcleaned = metrics.metrics.snr(clean_input_test_vec, filtered_Signal)
# #covnoisyclean= np.cov(noisy_input_test, pure_input_test)
# #covcleanedclean = np.cov(reconstruction,pure_input_test)
#
# rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
# rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)
#
# #plt.figure()
# #plt.plot(filteredsignal)
# #plt.figure()
# #plt.plot(noisy_input_vec)
#
# fm = open(file_path, 'a')
# fm.write("Filtred signal with BW filter 70Hz\n")
# fm.write("SNRNoisy: %f\n" % snrnoisy);
# fm.write("SNRCleaned: %f\n" % snrcleaned);
# fm.write("RMSNoisy: %f\n" % rmsNoisy);
# fm.write("RMSCleaned: %f\n" % rmsCleaned);
# fm.write("RMSENoisy: %f\n" % rrmseNoisy);
# fm.write("RMSECleaned: %f\n" % rrmseCleaned);
# fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
# fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
#
#
# fm.close()


# #computing snr for one sample
# print(len(result))
# for i in range(0, 6):
#     print("instance: ", i)
#     snr_on_sample = metrics.metrics.snr(result[i], clean_test[i])
#     print(snr_on_sample)
#     snr_on_sample = metrics.metrics.snr(filteredSignal[i], clean_test[i])
#     print(snr_on_sample)
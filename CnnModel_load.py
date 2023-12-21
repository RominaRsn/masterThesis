from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch
from masterThesis.metrics import metrics
from keras.utils import plot_model
import os
import masterThesis.model as models
import tensorflow as tf
import masterThesis.metrics
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
import masterThesis.metrics as metrics

data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")




# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)
#
#
smaller_noisy_train = noisy_train[0:1000]
smaller_clean_train = clean_train[0:1000]
#
#
# model = models.deep_CNN()
model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN.h5")
result = model.predict(noisy_test)
np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\result_deep_CNN", result)
#result = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN.npy")
#
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
# # Define the checkpoint callback
# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\checkPoint_cnn.h5'
# checkpoint = ModelCheckpoint(checkpoint_path,
#                              monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
#                              save_best_only=True,  # Save only if the validation performance improves
#                              mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
#                              verbose=1)  # Show messages about the checkpointing process
#
#
#
# model.optimizer.learning_rate = 1e-4
# model.fit(
#     noisy_train,
#     clean_train,
#     epochs=4,
#     batch_size=16,
#     validation_split=0.1,
#     callbacks=[callback, checkpoint],
#     shuffle=True
# )
#
#
# #result = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\result_cnn_k1.npy")
#
# result = model.predict(noisy_test)
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\result_cnn_k1.npy", result)
# result = model.predict(noisy_train[0:100])
# plt.subplot(3, 1, 1)
# plt.plot(result[10, :, :], label='result')
# plt.plot(clean_train[10, :], label='clean')
# plt.plot(noisy_train[10, :], label='noisy')
# plt.legend()
# plt.title('Comparison at Index 10')
#
# # Plot for index 100
# plt.subplot(3, 1, 2)
# plt.plot(result[50, :, :], label='result')
# plt.plot(clean_train[50, :], label='clean')
# plt.plot(noisy_train[50, :], label='noisy')
# plt.legend()
# plt.title('Comparison at Index 100')
#
# # Plot for index 100 again
# plt.subplot(3, 1, 3)
# plt.plot(result[70, :, :], label='result')
# plt.plot(clean_train[70, :], label='clean')
# plt.plot(noisy_train[70, :], label='noisy')
# plt.legend()
# plt.title('Comparison at Index 100 (again)')
#
# plt.tight_layout()  # Adjust layout for better spacing
# plt.show()




#
# model.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN_bigger_kernel.h5")
# model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN_bigger_kernel.h5")
# result = model.predict(noisy_test[0:100])
# # np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\deep_CNN_bigger_kernel_result.npy", result)
# model_cnn = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN.h5")
# result_cnn = model_cnn.predict(noisy_test[0:100])
# # np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\deep_CNN_result.npy", result_cnn)
# # #np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\results\deep_CNN_result.npy", result)
# model_encoder = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_more_filters.h5")
# result_encoder = model_encoder.predict(noisy_test[0:100, :])
# # plt.plot(result[10, :,:])
# # plt.plot(clean_train[10, :])
# # plt.plot(noisy_train[10, :])
# # plt.legend(['result', 'clean', 'noisy'])
# #
# # plt.show()
# #
# #
# # plt.plot(result[100, :,:])
# # plt.plot(clean_train[100, :])
# # plt.plot(noisy_train[100, :])
# # plt.legend(['result', 'clean', 'noisy'])
# #
# # plt.show()
# #
# #
# # plt.plot(result[100, :,:])
# # plt.plot(clean_train[100, :])
# # plt.plot(noisy_train[100, :])
# # plt.legend(['result', 'clean', 'noisy'])
# #
# #sharpening
# sharpening_kernel = np.array([0, -1, 2, -1, 0])
# #sharpening_kernel = np.array([0, -2, 3, -2, 0]) #intensified, not very good not very bad
# # Apply the kernel to the signal
# sharpened_signal = convolve1d(result[0, :], weights=sharpening_kernel, mode='constant', cval=0.0)
# ##plotting
# signalIndexVector = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
# #
# for i in signalIndexVector:
#     fig, axes = plt.subplots(nrows=3, ncols=1, sharey='col')
#
#     row_index = i
#     #row_index = np.random.randint(0, a)
#     #col_index = np.random.randint(0, 11520000/500)
#
#     axes[0].plot(clean_test[row_index, :], label = 'Clean Data')
#     axes[0].set_title('Clean data')
#     axes[0].set_ylabel('Signal amplitude')
#     axes[0].set_xlabel('Time')
#
#     #print(smaller_reshaped_data_clean_test[row_index, :].shape)
#
#
#     axes[1].plot(noisy_test[row_index, :], label = 'Noisy Data')
#     axes[1].set_title('Noisy data')
#     axes[1].set_ylabel('Signal amplitude')
#     axes[1].set_xlabel('Time')
#
#     #result = model.predict(result)
#     #result = result.transpose()
#
#     axes[2].plot(result[row_index, :], label='predicted data with CNN- big filter')
#     #axes[2].plot(convolve1d(result[row_index, :], weights=sharpening_kernel, mode='constant', cval=0.0), label='sharpened predicted data with CNN- big filter')
#     axes[2].set_title('predicted data with CNN- big filter')
#     axes[2].set_ylabel('Signal amplitude')
#     axes[2].set_xlabel('Time')
#     plt.legend()
#
#
#     # axes[3].plot(result_encoder[row_index, :], label='predicted data with AE')
#     # axes[3].plot(convolve1d(result_encoder[row_index, :], weights=sharpening_kernel, mode='constant', cval=0.0), label='sharpened predicted data with AE')
#     # axes[3].set_title('predicted data with AE')
#     # axes[3].set_ylabel('Signal amplitude')
#     # axes[3].set_xlabel('Time')
#     # plt.legend()
#     #
#     # axes[4].plot(result_cnn[row_index, :], label='predicted data with CNN')
#     # axes[4].plot(convolve1d(result_cnn[row_index, :], weights=sharpening_kernel, mode='constant', cval=0.0), label='sharpened predicted data with CNN')
#     # axes[4].set_title('predicted data with CNN')
#     # axes[4].set_ylabel('Signal amplitude')
#     # axes[4].set_xlabel('Time')
#
#     plt.legend()
#     plt.show()

#Calculating the metrics
clean_input_test_vec = np.ravel(clean_test)
noisy_input_test_vec = np.ravel(noisy_test)
test_reconstructions_vec = np.ravel(result)
cornoisyclean = np.corrcoef(clean_input_test_vec, noisy_input_test_vec)
corcleaned = np.corrcoef(clean_input_test_vec, test_reconstructions_vec)

snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)

snrcleaned = metrics.metrics.snr(clean_input_test_vec, test_reconstructions_vec)

rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, test_reconstructions_vec)


#compute rmse between noisy and clean test data
diffNoisyClean = noisy_test - clean_test
rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))

#compute rmse bwtween cleaned and clean test data
test_reconstructions = result.reshape((result.shape[0], result.shape[1]))
diffCleanedClean = test_reconstructions - clean_test
rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))

# Get the user's home directory
user_home = os.path.expanduser("~")
# Specify the file path in the Downloads directory
file_path = os.path.join(user_home, "Downloads", "cnn_k7.txt")

fm = open(file_path, 'w')
fm.write("Filtred signal with Cnn- kernelSize = 3\n")
fm.write("SNRNoisy: %f\n" % snrnoisy);
fm.write("SNRCleaned: %f\n" % snrcleaned);
fm.write("RMSNoisy: %f\n" % rmsNoisy);
fm.write("RMSCleaned: %f\n" % rmsCleaned);
fm.write("RMSENoisy: %f\n" % rrmseNoisy);
fm.write("RMSECleaned: %f\n" % rrmseCleaned);
fm.write("PearsonCorrNoisy: %f\n" % cornoisyclean[0, 1]);
fm.write("PearsonCorrCleaned: %f\n" % corcleaned[0, 1]);
fm.close()



# # Plot the results
# plt.figure(figsize=(10, 4))
# plt.plot(result[0, :], label='Original Signal')
# plt.plot(sharpened_signal, label='Sharpened Signal', linestyle='dashed')
# #plt.plot(noisy_test[0, :], label='noisy Signal', linestyle='dotted')
# plt.plot(clean_test[0, :], label='clean Signal', linestyle='dashdot')
# plt.legend()
# plt.title('1D Signal Sharpening')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.show()
# # Plot the results
# plt.figure(figsize=(10, 4))
# plt.plot(result[10, :], label='Original Signal')
# plt.plot(sharpened_signal, label='Sharpened Signal', linestyle='dashed')
# #plt.plot(noisy_test[10, :], label='noisy Signal', linestyle='dotted')
# plt.plot(clean_test[10, :], label='clean Signal', linestyle='dashdot')
# plt.legend()
# plt.title('1D Signal Sharpening')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.show()
# # Plot the results
# plt.figure(figsize=(10, 4))
# plt.plot(result[50, :], label='Original Signal')
# plt.plot(sharpened_signal, label='Sharpened Signal', linestyle='dashed')
# #plt.plot(noisy_test[50, :], label='noisy Signal', linestyle='dotted')
# plt.plot(clean_test[50, :], label='clean Signal', linestyle='dashdot')
# plt.legend()
# plt.title('1D Signal Sharpening')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.show()
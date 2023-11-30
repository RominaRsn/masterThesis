import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import metrics as metrics
import keras
from keras.models import load_model
import neurokit2 as nk

data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")




# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)



#result_3layer = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_three_layers_result_moreEpoch.npy")
result_3layer_more_filters = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_three_layers_result_morefilters.npy")
# result_4layer = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_four_layers_results.npy")
# result_5layer = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_results.npy")

result_5layer_more_filter = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_results_more_filters.npy")

filteredSignal_45 = nk.signal_filter(noisy_test, sampling_rate=250, lowcut=0.1, highcut=45, method='butterworth', order=4)
filteredSignal_70 = nk.signal_filter(noisy_test, sampling_rate=250, lowcut=0.1, highcut=70, method='butterworth', order=4)
filteredSignal_30 = nk.signal_filter(noisy_test, sampling_rate=250, lowcut=0.1, highcut=30, method='butterworth', order=4)

#, sharey='col'
# signalIndexVector = [0, 1, 3, 7, 11, 13, 14, 16, 17]
# for i in signalIndexVector:
#     fig, axes = plt.subplots(nrows=5 , ncols=1)
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
#     axes[2].plot(result_3layer_more_filters[row_index, :] - .55, label='predicted data with 3 layers in encoder')
#     axes[2].set_title('predicted data with 3 layers in encoder')
#     axes[2].set_ylabel('Signal amplitude')
#     axes[2].set_xlabel('Time')
#
#
#     # axes[3].plot(result_4layer[row_index, :] - .55, label ='predicted data with 4 layers in encoder')
#     # axes[3].set_title('predicted data with 4 layers in encoder')
#     # axes[3].set_ylabel('Signal amplitude')
#     # axes[3].set_xlabel('Time')
#
#     # axes[3].plot(result_5layer[row_index, :] - .55, label ='predicted data with 5 layers in encoder')
#     # axes[3].set_title('predicted data with 5 layers in encoder')
#     # axes[3].set_ylabel('Signal amplitude')
#     # axes[3].set_xlabel('Time')
#
#     axes[4].plot(result_5layer_more_filter[row_index, :], label ='predicted data with 5 layers in encoder with more filters')
#     axes[4].set_title('predicted data with 5 layers in encoder with more filters')
#     axes[4].set_ylabel('Signal amplitude')
#     axes[4].set_xlabel('Time')
#
#
#
#     #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
#     #print(test_array.shape())
#
#     # Add overall title
#     fig.suptitle('Comparison of different encoders', fontsize=16)
#
#     # Adjust layout to prevent overlap
#     #plt.tight_layout()
#
#     # Show the plot
#     plt.show()




# # Get the user's home directory
user_home = os.path.expanduser("~")

# Specify the file path in the Downloads directory
file_path = os.path.join(user_home, "Downloads", "your_file_bw.txt")


#3 layers

# result = result_3layer_more_filters
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
# file_path = os.path.join(user_home, "Downloads", "your_file_3layer.txt")
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


# ## 5 layers
# # model = load_model('model_with_five_layers_more_filters.h5')
# # result = model.predict(noisy_test)
# # np.save('result_5layer_more_filter.npy', result)
# result = np.load(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\result_5layer_more_filter.npy')
# #Calculating the metrics
clean_input_test_vec = np.ravel(clean_test)
noisy_input_test_vec = np.ravel(noisy_test)
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
diffNoisyClean = noisy_test - clean_test
rmsNoisy = np.sqrt(np.mean(diffNoisyClean**2))
#
# #compute rmse bwtween cleaned and clean test data
# test_reconstructions = result.reshape((result.shape[0], result.shape[1]))
# diffCleanedClean = test_reconstructions - clean_test
# rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))
#
# # Get the user's home directory
# user_home = os.path.expanduser("~")
# # Specify the file path in the Downloads directory
# file_path = os.path.join(user_home, "Downloads", "your_file_5layer.txt")
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


filtered_Signal = np.ravel(filteredSignal_45)

#compute rmse bwtween cleaned and clean test data
diffCleanedClean = filtered_Signal-clean_input_test_vec
rmsCleaned = np.sqrt(np.mean(diffCleanedClean**2))

cornoisyclean = np.corrcoef(clean_input_test_vec , noisy_input_test_vec)
corcleaned = np.corrcoef(clean_input_test_vec , filtered_Signal)
snrnoisy = metrics.metrics.snr(clean_input_test_vec, noisy_input_test_vec)
snrcleaned = metrics.metrics.snr(clean_input_test_vec, filtered_Signal)
#covnoisyclean= np.cov(noisy_input_test, pure_input_test)
#covcleanedclean = np.cov(reconstruction,pure_input_test)

rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)

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
#covnoisyclean= np.cov(noisy_input_test, pure_input_test)
#covcleanedclean = np.cov(reconstruction,pure_input_test)

rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)

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
#covnoisyclean= np.cov(noisy_input_test, pure_input_test)
#covcleanedclean = np.cov(reconstruction,pure_input_test)

rrmseNoisy = metrics.metrics.rrmseMetric(clean_input_test_vec, noisy_input_test_vec)
rrmseCleaned = metrics.metrics.rrmseMetric(clean_input_test_vec, filtered_Signal)

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


fm.close()


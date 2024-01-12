import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import convolve1d
import neurokit2 as nk
import scipy
from sklearn.metrics import confusion_matrix
import keras
from keras.models import load_model
from statistics import  mean

from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
import masterThesis.model as model
from keras import layers, models, optimizers


#
# combo_model_result = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cnn_combo\result\result_pat_14_sz_2_ch_1.npy")
# combo_model_result = combo_model_result.squeeze(-1)
#
# data = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\pat_14_sz_2_ch_1.npy")
# max_clean = np.max(data)
# min_clean = np.min(data)
# data_clean_normalized = (data - min_clean) / (max_clean - min_clean)
# data_clean_normalized = data_clean_normalized - np.average(data_clean_normalized)
#
# fig, axes = plt.subplots(nrows=2, ncols=1, sharey='col')
# i = 1000
# #row_index = np.random.randint(0, a)
# #col_index = np.random.randint(0, 11520000/500)
#
# axes[0].plot(data_clean_normalized[i, :], label = 'Real Data')
# axes[0].set_title('Real data')
# axes[0].set_ylabel('Signal amplitude')
# axes[0].set_xlabel('Time')
#
# #print(smaller_reshaped_data_clean_test[row_index, :].shape)
#
#
# axes[1].plot(combo_model_result[i, :], label = 'cleaned Data')
# axes[1].set_title('cleaned data')
# axes[1].set_ylabel('Signal amplitude')
# axes[1].set_xlabel('Time')
#
# plt.legend()
# plt.show()


def classifiedAtLeastOnce(true_labels, predicted_labels):
    true_index = np.where(true_labels == 1)
    predicted_index = np.where(predicted_labels == 1)

    if np.intersect1d(true_index, predicted_index).size > 0:
        return 1
    else:
        return 0








folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"
def countNumberOfSeizuresPerPerson(patient_number):
    # Iterate through all files in the folder
    cnt = 0
    for filename in os.listdir(folder_path):
        # Check if the file starts with "pat_1" and has the ".npy" extension
        if filename.startswith(f"pat_{patient_number}_") and filename.endswith(f"ch_1.npy"):
            # Construct the full path to the file

            cnt += 1
    return cnt

def sensitivity(generated_labels , true_labels):
    TP = 0
    FN = 0
    if(generated_labels.shape[0] != true_labels.shape[0]):  # check if the number of labels is the same
        print("The number of labels is not the same")
        return
    else:
        true_labels_size = true_labels.shape[0]
        for i in range(0, true_labels_size):
            if generated_labels[i] == 1 and true_labels[i] == 1:
                TP += 1
            if generated_labels[i] == 0 and true_labels[i] == 1:
                FN += 1

            #"sens": TP/(TP+FN) ,
        a = {"TP": TP, "FN": FN}
        return a

def specificity(generated_labels , true_labels):
    TN = 0
    FP = 0
    if(generated_labels.shape[0] != true_labels.shape[0]):  # check if the number of labels is the same
        print("The number of labels is not the same")
        return
    else:
        true_labels_size = true_labels.shape[0]

        for i in range(0, true_labels_size):
            if generated_labels[i] == 0 and true_labels[i] == 0:
                TN += 1
            if generated_labels[i] == 1 and true_labels[i] == 0:
                FP += 1

            #"spec": TN/(TN+FP) ,
        a = {"TN": TN, "FP": FP}
        return a

def normalize_ch_data(data1, data2, data3, data4):

    if np.any(np.isnan(data1)) or np.any(np.isnan(data2)) or np.any(np.isnan(data3)) or np.any(np.isnan(data4)):
        raise ValueError("Input data contains NaN values.")

    if np.any(np.isinf(data1)) or np.any(np.isinf(data2)) or np.any(np.isinf(data3)) or np.any(np.isinf(data4)):
        raise ValueError("Input data contains Inf values.")

    max_val = np.max([np.max(data1), np.max(data2), np.max(data3), np.max(data4)])
    min_val = np.min([np.min(data1), np.min(data2), np.min(data3), np.min(data4)])

    #avg_val = np.average(np.average(data1) + np.average(data2) + np.average(data3) + np.average(data4))

    #avg_val = np.average(data1 + data2 + data3 + data4)

    data1 = (data1 - min_val) / (max_val - min_val)
    #data1 = data1 - avg_val

    data2 = (data2 - min_val) / (max_val - min_val)
    #data2 = data2 - avg_val

    data3 = (data3 - min_val) / (max_val - min_val)
    #data3 = data3 - avg_val

    data4 = (data4 - min_val) / (max_val - min_val)
    #data4 = data4 - avg_val

    print("max_val: ", max_val)
    print("min_val: ", min_val)

    print("data1 max: ", np.max(data1))
    print("data1 min: ", np.min(data1))
    print("data1 avg: ", np.average(data1))

    print("data2 max: ", np.max(data2))
    print("data2 min: ", np.min(data2))
    print("data2 avg: ", np.average(data2))

    print("data3 max: ", np.max(data3))
    print("data3 min: ", np.min(data3))
    print("data3 avg: ", np.average(data3))


    return data1, data2, data3, data4


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def thetaBandPower(epoched_data):
    sampling_freq = 250
    f, power_block = scipy.signal.welch(epoched_data, fs=sampling_freq, window='hann', axis=1,
                                        scaling='spectrum')  # ----------------# ERROR source at 1sec window, non-overlaping -> nperseg=None
    thetaband_st = find_nearest(f, 4)[0]
    # thetaband_end = int((np.where(f == 8)[0])[0])
    thetaband_end = find_nearest(f, 8)[0]
    thetaband_power = power_block[:, thetaband_st:thetaband_end + 1].sum(axis=1)[:, None]
    return thetaband_power


def linelength(data):
    data_diff = np.diff(data)
    return np.sum(np.absolute(data_diff), axis=1)

path_extension_labels = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s"

sharpening_kernel = np.array([0, -1, 2, -1, 0])
def sharpenSignal(data):
    data = np.array(data)
    data_sharpened = np.empty_like(data)
    for i in range(0, len(data)):
        data_sharpened[i, :] = convolve1d(data[i, :], weights=sharpening_kernel, mode='constant', cval=0.0)
    return data_sharpened

def countSeizures(true_labels, predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4):
    true_index = np.where(true_labels == 1)
    true_index = np.array(true_index)

    predicted_index_1 = np.where(predicted_labels_1 == 1)
    predicted_index_1 = np.array(predicted_index_1)

    predicted_index_2 = np.where(predicted_labels_2 == 1)
    predicted_index_2 = np.array(predicted_index_2)

    predicted_index_3 = np.where(predicted_labels_3 == 1)
    predicted_index_3 = np.array(predicted_index_3)

    predicted_index_4 = np.where(predicted_labels_4 == 1)
    predicted_index_4 = np.array(predicted_index_4)

    





auc_list_new = []
auc_list_old = []
auc_list_bandpass = []
auc_list_30 = []
auc_list_45 = []
auc_list_70 = []

TP_list_new = []
FP_list_new = []

TP_list_old = []
FP_list_old = []

TP_list_lowpass = []
FP_list_lowpass = []

TN_list_new = []
FN_list_new = []

TN_list_old = []
FN_list_old = []

cm_45_list = []
cm_raw_list = []
cm_predicted_list = []

classified_as_sz_list_45 = []
classified_as_sz_list_raw = []
classified_as_sz_list_predicted = []


#model_2 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_more_filters.h5')
#model_2 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN_bigger_kernel.h5')
#model_2 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_checkpoint.h5')


for p in range(1, 51):
    sz_num = countNumberOfSeizuresPerPerson(p)

    for i in range(1, sz_num + 1):
        for ch in range(1, 5):
            file_path = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_{ch}.npy")
            data = np.load(file_path)
            file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{i}_labels.npy")

            label = np.load(file_path_labels)

            # file_path_1 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_1.npy")
            # file_path_2 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_2.npy")
            # file_path_3 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_3.npy")
            # file_path_4 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_4.npy")
            #
            # data_1 = np.load(file_path_1)
            # data_2 = np.load(file_path_2)
            # data_3 = np.load(file_path_3)
            # data_4 = np.load(file_path_4)
            #
            # data_1, data_2, data_3, data_4 = normalize_ch_data(data_1, data_2, data_3, data_4)
            #
            # data = np.empty_like(data_1)
            # if (ch == 1):
            #     data = data_1
            # elif (ch == 2):
            #     data = data_2
            # elif (ch == 3):
            #     data = data_3
            # elif (ch == 4):
            #     data = data_4

            mean_val = np.mean(data)
            std_val = np.std(data)
            # Normalize the data to the range [-1, 1]
            new_normalized_data = (data - mean_val) / std_val
            new_normalized_data = (new_normalized_data) / (np.max(new_normalized_data) - np.min(new_normalized_data))



            #num_zeros = (0, 12)

            # Pad the array with zeros
            #padded_data = np.pad(data, ((0, 0), num_zeros), mode='constant')

            # predicted_data = model_2.predict(new_normalized_data)
            # predicted_data = predicted_data.squeeze(-1)
            # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_norm_method", f"pat_{p}_sz_{i}_ch_{ch}.npy"), predicted_data)
            #np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\CNN\result_cnn", f"pat_{p}_sz_{i}_ch_{ch}.npy"), predicted_data)
            #np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\GRU_cheby", f"pat_{p}_sz_{i}_ch_{ch}.npy"), predicted_data)
            #predicted_data = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\CNN\result_cnn", f"pat_{p}_sz_{i}_ch_{ch}.npy"))
            #predicted_data = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby", f"pat_{p}_sz_{i}_ch_{ch}.npy"))
            #predicted_data = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_norm_method", f"pat_{p}_sz_{i}_ch_{ch}.npy"))
            predicted_data = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\GRU_cheby", f"pat_{p}_sz_{i}_ch_{ch}.npy"))
            # predicted_data = model_2.predict(padded_data)
            # predicted_data = predicted_data.squeeze(-1)
            #predicted_data = sharpenSignal(predicted_data)

            filteredSignal_45 = nk.signal_filter(data, sampling_rate=250, highcut=40,
                                               method='butterworth', order=4)
            # filteredSignal_70 = nk.signal_filter(data, sampling_rate=250, lowcut=0.1, highcut=70,
            #                                      method='butterworth', order=4)
            # filteredSignal_30 = nk.signal_filter(data, sampling_rate=250, lowcut=0.1, highcut=30,
            #                                      method='butterworth', order=4)


            new_ll = thetaBandPower(predicted_data)
            old_ll = thetaBandPower(data)
            # ll_30 = linelength(filteredSignal_30)
            ll_45 = thetaBandPower(filteredSignal_45)
            # ll_70 = linelength(filteredSignal_70)

            # new_ll = thetaBandPower(predicted_data)
            # old_ll = thetaBandPower(data)
            # # ll_30 = thetaBandPower(filteredSignal_30)
            # ll_45 = thetaBandPower(filteredSignal_45)
            # ll_70 = thetaBandPower(filteredSignal_70)



            # avg_30 = np.average(ll_30)
            # std_30 = np.std(ll_30)
            # thresholds_30 = [avg_30 - 3 * std_30, avg_30 - 2 * std_30, avg_30 - std_30, avg_30, avg_30 + std_30, avg_30 + 2 * std_30, avg_30 + 3 * std_30]
            #
            #
            # sens_30 = []
            # spec_30 = []
            # for th in thresholds_30:
            #     new_ll_label = (ll_30 > th).astype(int)
            #     sens1 = sensitivity(new_ll_label, label)
            #     spec1 = specificity(new_ll_label, label)
            #     sens_30.append(sens1["sens"])
            #     spec_30.append(1 - spec1["spec"])

            avg_45 = np.average(ll_45)
            std_45 = np.std(ll_45)
            thresholds_45 = [avg_45 - 3 * std_45, avg_45 - 2 * std_45, avg_45 - std_45, avg_45, avg_45 + std_45, avg_45 + 2 * std_45, avg_45 + 3 * std_45]

            sens_45 = []
            spec_45 = []
            cm_45 = []  # confusion matrix
            classification_list_45 = []
            for th in thresholds_45:
                tn, fp, fn, tp = confusion_matrix(label, (ll_45 > th).astype(int)).ravel()
                cm_45.append([tn, fp, fn, tp])
                new_ll_label = (ll_45 > th).astype(int)
                classification_list_45.append(classifiedAtLeastOnce(label, new_ll_label))
                sens1 = sensitivity(new_ll_label, label)
                spec1 = specificity(new_ll_label, label)
                sens_45.append(sens1["TP"])
                spec_45.append(spec1["FP"])

            TP_list_lowpass.append(mean(sens_45))
            FP_list_lowpass.append(mean(spec_45))
            cm_45_list.append(cm_45)
            classified_as_sz_list_45.append(classification_list_45)

            # avg_70 = np.average(ll_70)
            # std_70 = np.std(ll_70)
            # thresholds_70 = [avg_70 - 3 * std_70, avg_70 - 2 * std_70, avg_70 - std_70, avg_70, avg_70 + std_70, avg_70 + 2 * std_70, avg_70 + 3 * std_70]
            #
            # sens_70 = []
            # spec_70 = []
            # for th in thresholds_70:
            #     new_ll_label = (ll_70 > th).astype(int)
            #     sens1 = sensitivity(new_ll_label, label)
            #     spec1 = specificity(new_ll_label, label)
            #     sens_70.append(sens1["sens"])
            #     spec_70.append(1 - spec1["spec"])
            #
            #

            old_avg = np.average(old_ll)
            old_std = np.std(old_ll)
            thresholds_old = [old_avg - 3 * old_std, old_avg - 2 * old_std, old_avg - old_std, old_avg,
                              old_avg + old_std, old_avg + 2 * old_std, old_avg + 3 * old_std]

            avg = np.average(new_ll)
            std = np.std(new_ll)
            thresholds = [avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std]

            sens_new = []
            spec_new = []
            cm_new = []  # confusion matrix
            classification_list_new = []
            for th in thresholds:
                tn, fp, fn, tp = confusion_matrix(label, (new_ll > th).astype(int)).ravel()
                cm_new.append([tn, fp, fn, tp])
                new_ll_label = (new_ll > th).astype(int)
                classification_list_new.append(classifiedAtLeastOnce(label, new_ll_label))
                sens1 = sensitivity(new_ll_label, label)
                spec1 = specificity(new_ll_label, label)
                sens_new.append(sens1["TP"])
                spec_new.append(spec1["FP"])

            TP_list_new.append(mean(sens_new))
            FP_list_new.append(mean(spec_new))
            cm_predicted_list.append(cm_new)
            classified_as_sz_list_predicted.append(classification_list_new)

            sens_old = []
            spec_old = []
            cm_old = []  # confusion matrix
            classification_list_raw = []
            for th in thresholds_old:
                tn, fp, fn, tp = confusion_matrix(label, (old_ll > th).astype(int)).ravel()
                cm_old.append([tn, fp, fn, tp])
                old_ll_label = (old_ll > th).astype(int)
                classification_list_raw.append(classifiedAtLeastOnce(label, old_ll_label))
                sens2 = sensitivity(old_ll_label, label)
                spec2 = specificity(old_ll_label, label)
                sens_old.append(sens2["TP"])
                spec_old.append(spec2["FP"])

            TP_list_old.append(mean(sens_old))
            FP_list_old.append(mean(spec_old))
            cm_raw_list.append(cm_old)
            classified_as_sz_list_raw.append(classification_list_raw)

            # print("false positive")
            # print(spec_new)
            # print("average FP new", mean(spec_new))
            # print(spec_old)
            # print("average FP old", mean(spec_old))
            #
            #
            # print("true positive")
            # print(sens_new)
            # print("average TP new", mean(sens_new))
            # print(sens_old)
            # print("average TP old", mean(sens_old))



#             auc_new = auc(spec_new, sens_new)
#             auc_old = auc(spec_old, sens_old)
#             # auc_30 = auc(spec_30, sens_30)
#             #auc_45 = auc(spec_45, sens_45)
#             # auc_70 = auc(spec_70, sens_70)
#
#             auc_list_new.append(auc_new)
#             auc_list_old.append(auc_old)
#             # auc_list_30.append(auc_30)
#             #auc_list_45.append(auc_45)
#             # auc_list_70.append(auc_70)
#
#
#             print(f"patient {p}, seizure {i}, channel {ch}, auc_new {auc_new}, auc_old {auc_old}")
#
#
# print(f"mean auc_new {np.mean(auc_list_new)}, mean auc_old {np.mean(auc_list_old)}")
#print(f"std auc_new {np.std(auc_list_new)}, std auc_old {np.std(auc_list_old)}")
#print(f"mean auc_30 {np.mean(auc_list_30)}, mean auc_45 {np.mean(auc_list_45)}, mean auc_70 {np.mean(auc_list_70)}")



classified_as_sz_list_45 = np.array(classified_as_sz_list_45)
print(classified_as_sz_list_45.shape)
classified_as_sz_list_predicted = np.array(classified_as_sz_list_predicted)
classified_as_sz_list_raw = np.array(classified_as_sz_list_raw)

average_values_45_list = np.mean(classified_as_sz_list_45, axis=0)
average_values_predicted_list = np.mean(classified_as_sz_list_predicted, axis=0)
average_values_raw_list = np.mean(classified_as_sz_list_raw, axis=0)

print(average_values_45_list)
print(average_values_predicted_list)
print(average_values_raw_list)

#
# # Convert the list to a numpy array for easier calculations
results_array_45 = np.array(cm_45_list)
#
# #results_array_45 = results_array_45.reshape(results_array_45.shape[0] * results_array_45.shape[1], 4)
# print(results_array_45.shape)
# # Calculate the average for each column (TN, FP, FN, TP)
average_values = np.mean(results_array_45, axis=0)

col_fp_45 = average_values[:, 1]
#col_tp_45 = average_values[:, 3]
# auc_45 = auc(col_fp_45, col_tp_45)
auc_45 = auc(col_fp_45, average_values_45_list)
#
#
results_array_new = np.array(cm_predicted_list)
# #results_array_new = results_array_new.reshape(results_array_new.shape[0] * results_array_new.shape[1], 4)
average_values_new = np.mean(results_array_new, axis=0)
col_fp_new = average_values_new[:, 1]
# col_tp_new = average_values_new[:, 3]
auc_new = auc(col_fp_new, average_values_predicted_list)
#
#
results_array_old = np.array(cm_raw_list)
# #results_array_old = results_array_old.reshape(results_array_old.shape[0] * results_array_old.shape[1], 4)
average_values_old = np.mean(results_array_old, axis=0)
col_fp_old = average_values_old[:, 1]
# col_tp_old = average_values_old[:, 3]
auc_old = auc(col_fp_old, average_values_raw_list)

plt.plot(col_fp_45, average_values_45_list, label='45')
plt.plot(col_fp_new, average_values_predicted_list, label='new')
plt.plot(col_fp_old, average_values_raw_list, label='old')
plt.legend(labels=[f'auc_old: {auc_old:.2f}', f'auc_new: {auc_new:.2f}', f'auc_45: {auc_45:.2f}'], loc='lower right')
plt.xlabel('FP')
plt.ylabel('classified as seizure at least once')
plt.show()



#
# plt.plot(col_fp_45, col_tp_45, label='45')
# plt.plot(col_fp_new, col_tp_new, label='new')
# plt.plot(col_fp_old, col_tp_old, label='old')
# plt.legend(labels=[f'auc_old: {auc_old:.2f}', f'auc_new: {auc_new:.2f}', f'auc_45: {auc_45:.2f}'], loc='lower right')
# plt.xlabel('FP')
# plt.ylabel('TP')
# plt.show()
#
#
# print('order of values: TN, FP, FN, TP')
# print("average values 45")
# print(average_values)
# print("average values new")
# print(average_values_new)
# print("average values old")
# print(average_values_old)



# print("normalize over channels")
# print(f"mean TP new {mean(TP_list_new)}, mean TP old {mean(TP_list_old)}, mean TP lowpass {mean(TP_list_lowpass)}")
# print(f"mean FP new {mean(FP_list_new)}, mean FP old {mean(FP_list_old)}, mean FP lowpass {mean(FP_list_lowpass)}")
#


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import convolve1d
import neurokit2 as nk
import scipy
from sklearn.metrics import confusion_matrix, recall_score
from collections import Counter
import keras
from keras.models import load_model
from statistics import mean
from sklearn.metrics import f1_score

from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
import masterThesis.model as model
from keras import layers, models, optimizers
import neurokit2 as nk
import cProfile
from memory_profiler import profile



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


# def classifiedAtLeastOnce(true_labels, predicted_labels):
#     true_index = np.where(true_labels == 1)
#     predicted_index = np.where(predicted_labels == 1)
#
#     # print(np.intersect1d(true_index, predicted_index))
#     # print("****************************************************************************************************")
#
#     if np.intersect1d(true_index, predicted_index).size > 0:
#         return 1
#     else:
#         return 0
#

# def getTheChannelNumberThatDetectsTheSeizure(true_labels, predicted_label_1, predicted_labe_2, predicted_label_3, predicted_label_4):
#     for i in range(0, predicted_label_1.shape[1]):
#         col_predic_1 = predicted_label_1[:, i]
#         col_predic_2 = predicted_labe_2[:, i]
#         col_predic_3 = predicted_label_3[:, i]
#         col_predic_4 = predicted_label_4[:, i]
#
#         concat_predic = np.concatenate([col_predic_1, col_predic_2, col_predic_3, col_predic_4])
#
#         res = np.logical_or(col_predic_1, col_predic_2)
#         res = np.logical_or(res, col_predic_3)
#         res = np.logical_or(res, col_predic_4)
#
#         index = np.where(res == 1)
#
#         for

def doLogicalOR(predicted_label_1, predicted_labe_2, predicted_label_3, predicted_label_4):
    result = np.empty_like(predicted_label_1)
    for i in range(0, predicted_label_1.shape[1]):
        col_predic_1 = predicted_label_1[:, i]
        col_predic_2 = predicted_labe_2[:, i]
        col_predic_3 = predicted_label_3[:, i]
        col_predic_4 = predicted_label_4[:, i]
        res = np.logical_or(col_predic_1, col_predic_2)
        res = np.logical_or(res, col_predic_3)
        res = np.logical_or(res, col_predic_4)
        result[:, i] = res
    return result


def getTheBestThresholds(true_labels, predicted_label):
    threshold_list = []
    for i in range(0,predicted_label.shape[1]):
        col_predic = predicted_label[:, i]
        sens = recall_score(true_labels, col_predic)
        if(sens > 0.85):
            threshold_list.append(i)

    #print(threshold_list)
    return threshold_list


def getNoneDetections(true_labels, predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4):

    true_index = np.where(true_labels == 1)

    false_detection_list= []


    for i in range(0, predicted_labels_1.shape[1]):
        pred_1_col = predicted_labels_1[:, i]
        pred_2_col = predicted_labels_2[:, i]
        pred_3_col = predicted_labels_3[:, i]
        pred_4_col = predicted_labels_4[:, i]

        pred_1_index = np.where(pred_1_col == 1)
        pred_2_index = np.where(pred_2_col == 1)
        pred_3_index = np.where(pred_3_col == 1)
        pred_4_index = np.where(pred_4_col == 1)

        # Combine all predicted indices into a single array
        all_pred_indices = np.concatenate([pred_1_index[0], pred_2_index[0], pred_3_index[0], pred_4_index[0]])

        # Find indices in true_index that are not in any of the pred_indices
        not_in_pred_indices = true_index[0][~np.in1d(true_index[0], all_pred_indices)]
        false_detection_list.append(not_in_pred_indices)

        # Combine all predicted indices into a single list
    all_not_pred_elements = [element for sublist in false_detection_list for element in sublist]

    all_not_pred_elements = np.unique(all_not_pred_elements)

    # Find elements in true_list that are not in any of the pred_elements

    return all_not_pred_elements

def getReducedFlaseDetections(true_label, predicted_label, label_from_raw_data, selected_threshold):

    true_index = np.where(true_label == 0)[0] #######negated true index

    # predicted_label = np.all(predicted_label, axis=1)
    # label_from_raw_data = np.all(label_from_raw_data, axis=1)
    # predicted_label = predicted_label.astype(int)
    # label_from_raw_data = label_from_raw_data.astype(int)

    predicted_label = predicted_label[:, selected_threshold]
    label_from_raw_data = label_from_raw_data[:, selected_threshold]

    predicted_index = np.where(predicted_label == 0)[0]
    negative_label_from_raw_data_index = np.where(label_from_raw_data == 1)[0]

    intersection = np.intersect1d(true_index, predicted_index)
    intersection = np.intersect1d(intersection, negative_label_from_raw_data_index)

    return intersection







def getFlaseDetections(true_label, predicted_label, label_from_raw_data, selected_threshold):
    # #true_index = np.where(true_labels == 0)
    # true_index = np.where(true_labels == 1)
    # false_detection_list = []
    # for i in range(0, predicted_label.shape[1]):
    #     list_inner = []
    #     pred_col = predicted_label[:, i]
    #     pred_index = np.where(pred_col == 1)
    #     # if np.intersect1d(true_index, pred_index).size > 0:
    #     #     false_detection_list.append(np.intersect1d(true_index, pred_index))
    #     for j in pred_index[0]:
    #         if j not in true_index[0]:
    #             list_inner.append(j)
    #     false_detection_list.append(list_inner)
    #
    # # Find common numbers
    # common_elements = false_detection_list[0]
    #
    # # Iterate over the rest of the arrays
    # for array in false_detection_list[1:]:
    #     common_elements = np.intersect1d(common_elements, array)

    true_index = np.where(true_label == 0)[0] #######negated true index



    # predicted_label = np.all(predicted_label, axis=1)
    # label_from_raw_data = np.all(label_from_raw_data, axis=1)
    # predicted_label = predicted_label.astype(int)
    # label_from_raw_data = label_from_raw_data.astype(int)

    predicted_label = predicted_label[:, selected_threshold]
    label_from_raw_data = label_from_raw_data[:, selected_threshold]

    predicted_index = np.where(predicted_label == 1)[0]
    negative_label_from_raw_data_index = np.where(label_from_raw_data == 1)[0]

    intersection = np.intersect1d(true_index, predicted_index)
    intersection = np.intersect1d(intersection, negative_label_from_raw_data_index)

    return intersection


def getdisImprovement(true_label, label_from_raw_data, predicted_label, selected_threshold):
    # #true_index = np.where(true_labels == 0)
    # true_index = np.where(true_labels == 1)
    # false_detection_list = []
    # for i in range(0, predicted_label.shape[1]):
    #     list_inner = []
    #     pred_col = predicted_label[:, i]
    #     pred_index = np.where(pred_col == 1)
    #     # if np.intersect1d(true_index, pred_index).size > 0:
    #     #     false_detection_list.append(np.intersect1d(true_index, pred_index))
    #     for j in pred_index[0]:
    #         if j not in true_index[0]:
    #             list_inner.append(j)
    #     false_detection_list.append(list_inner)
    #
    # # Find common numbers
    # common_elements = false_detection_list[0]
    #
    # # Iterate over the rest of the arrays
    # for array in false_detection_list[1:]:
    #     common_elements = np.intersect1d(common_elements, array)

    true_index = np.where(true_label == 1)[0] #######negated true index

    # predicted_label = np.all(predicted_label, axis=1)
    # label_from_raw_data = np.all(label_from_raw_data, axis=1)
    #
    # predicted_label = predicted_label.astype(int)
    # label_from_raw_data = label_from_raw_data.astype(int)

    predicted_label = predicted_label[:, selected_threshold]
    label_from_raw_data = label_from_raw_data[:, selected_threshold]

    predicted_index = np.where(predicted_label == 0)[0]
    negative_label_from_raw_data_index = np.where(label_from_raw_data == 1)[0]

    improved_label = np.zeros(len(true_label))
    for j in range(0, len(true_label)):
        if label[j] == 1 and predicted_label[j] == 0 and label_from_raw_data[j] == 1:
            improved_label[j] = 1

    intersection = np.where(improved_label == 1)[0]
    return intersection

    # intersection = np.intersect1d(true_index, predicted_index)
    # intersection = np.intersect1d(intersection, negative_label_from_raw_data_index)

    return intersection


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

def sensitivity(confusion_matrix):
    TP = confusion_matrix["TP"]
    FN = confusion_matrix["FN"]
    return TP/(TP+FN)

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


def seizureClassification(true_labels, predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4):
    # true_index = np.where(true_labels == 1)
    # true_index = np.array(true_index)
    #
    # predicted_index_1 = np.where(predicted_labels_1 == 1)
    # predicted_index_1 = np.array(predicted_index_1)
    #
    # predicted_index_2 = np.where(predicted_labels_2 == 1)
    # predicted_index_2 = np.array(predicted_index_2)
    #
    # predicted_index_3 = np.where(predicted_labels_3 == 1)
    # predicted_index_3 = np.array(predicted_index_3)
    #
    # predicted_index_4 = np.where(predicted_labels_4 == 1)
    # predicted_index_4 = np.array(predicted_index_4)

    if(np.array_equal(predicted_labels_1, predicted_labels_2) and np.array_equal(predicted_labels_2, predicted_labels_3) and np.array_equal(predicted_labels_3, predicted_labels_4)):
        print("all equal")

    predicted_label_each_threshold = []
#each threshold
    for i in range(0, predicted_labels_1.shape[1]):
        mat_1 = predicted_labels_1[:, i]
        mat_2 = predicted_labels_2[:, i]
        mat_3 = predicted_labels_3[:, i]
        mat_4 = predicted_labels_4[:, i]

        #print(mat_1 + mat_2 + mat_3 + mat_4)

        predicted_label = np.empty_like(true_labels)
        for j in range(0, len(mat_1)):
            if mat_1[j] == 1 or mat_2[j] == 1 or mat_3[j] == 1 or mat_4[j] == 1:
                predicted_label[j] = 1
            else:
                predicted_label[j] = 0
                #print("negative")
        predicted_label_each_threshold.append(predicted_label)

    predicted_label_each_threshold = np.array(predicted_label_each_threshold)
    predicted_label_each_threshold = predicted_label_each_threshold.T
    #if(np.array_equal(predicted_label_each_threshold[:, 1], predicted_label_each_threshold[:, 6])):
        #print("equal")
    conf_list = []
    for i in range(0, predicted_label_each_threshold.shape[1]):
        conf_matrix = confusion_matrix(true_labels, predicted_label_each_threshold[:, i]).ravel()
        conf_list.append(conf_matrix)



    return conf_list


def classifiedAtLeastOnce(true_labels, predicted_labels):
    return_list = []
    for i in range(0, predicted_labels.shape[1]):
        threshold_predicted = predicted_labels[:, i]
        true_index = np.where(true_labels == 1)
        predicted_index = np.where(threshold_predicted == 1)
        if np.intersect1d(true_index, predicted_index).size > 0:
            return_list.append(1)
        else:
            return_list.append(0)
    return return_list
    # print(np.intersect1d(true_index, predicted_index))
    # print("****************************************************************************************************")




def classifiedAtLeastOnce_10sec(true_labels, predicted_labels):
    # true_index = np.where(true_labels == 1)
    # predicted_index = np.where(predicted_labels == 1)
    # predicted_index = np.array(predicted_index)
    #
    # first = true_index[0][0]
    # list_indexes = [first, first + 1, first + 2, first + 3, first + 4, first + 5]
    # list_indexes = np.array(list_indexes)
    #
    # in_list = np.isin(list_indexes, predicted_index)
    #
    # if(in_list.any()):
    #     return 1
    # else:
    #     return 0
    return_list = []
    for i in range(0, predicted_labels.shape[1]):
        threshold_predicted = predicted_labels[:, i]
        true_index = np.where(true_labels == 1)
        predicted_index = np.where(threshold_predicted == 1)
        first = true_index[0][0]
        list_indexes = [first, first + 1, first + 2, first + 3, first + 4, first + 5]
        list_indexes = np.array(list_indexes)

        in_list = np.isin(list_indexes, predicted_index)

        if(in_list.any()):
            return_list.append(1)
        else:
            return_list.append(0)
    return return_list


def getOnlyLabels(data, true_labels, thresholds):

    ll = linelength(data)
    # #ll = thetaBandPower(data)
    #
    # avg = np.average(ll)
    # std = np.std(ll)
    # thresholds = [avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std]

    new_label = []
    classifed = []
    for th in thresholds:
        new_label.append((ll > th).astype(int))
        #classifed.append(classifiedAtLeastOnce(true_labels, new_label[-1]))
        #classifed.append(classifiedAtLeastOnce_10sec(true_labels, new_label[-1]))
    array = np.array(new_label)
    array = array.T
    return array


def getLabels(data, true_labels):

    ll = linelength(data)
    #ll = thetaBandPower(data)

    avg = np.average(ll)
    std = np.std(ll)
    thresholds = [avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std]

    new_label = []
    classifed = []
    for th in thresholds:
        new_label.append((ll > th).astype(int))
        #classifed.append(classifiedAtLeastOnce(true_labels, new_label[-1]))
        classifed.append(classifiedAtLeastOnce_10sec(true_labels, new_label[-1]))
    array = np.array(new_label)
    array = array.T
    return array, np.array(classifed)

def plotFalsePositives(index_list, data_1, data_2, data_3, data_4, predictions_1, predictions_2, predictions_3, predictions_4, patient_number, seizure_number, selected_threshold, predicted_label_1, predicted_label_2, predicted_label_3, predicted_label_4):

    predicted_label_col_1 = predicted_label_1[:, selected_threshold]
    predicted_label_col_2 = predicted_label_2[:, selected_threshold]
    predicted_label_col_3 = predicted_label_3[:, selected_threshold]
    predicted_label_col_4 = predicted_label_4[:, selected_threshold]

    predicted_ones_1 = np.where(predicted_label_col_1 == 1)[0]
    predicted_ones_2 = np.where(predicted_label_col_2 == 1)[0]
    predicted_ones_3 = np.where(predicted_label_col_3 == 1)[0]
    predicted_ones_4 = np.where(predicted_label_col_4 == 1)[0]


    for i in index_list:

        if(i>10):
            whichChannelHasSeizure = [0, 0, 0, 0]
            if(i in predicted_ones_1):
                whichChannelHasSeizure[0] = 1
            if(i in predicted_ones_2):
                whichChannelHasSeizure[1] = 1
            if(i in predicted_ones_3):
                whichChannelHasSeizure[2] = 1
            if(i in predicted_ones_4):
                whichChannelHasSeizure[3] = 1


            # Create subplots with specified axes
            fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharey='col')
            fig.suptitle(f'patient: {patient_number}, seizure: {seizure_number}, index: {i}')
            # Plot each subplot
            axes[0].plot(data_1[i - 5:i + 5, :].ravel(), label='Data 1')
            axes[0].plot(predictions_1[i - 5:i + 5, :].ravel(), label='Predictions 1')
            axes[0].legend()
            if(whichChannelHasSeizure[0] == 1):
                axes[0].set_title("Seizure in this channel")

            axes[1].plot(data_2[i - 5:i + 5, :].ravel(), label='Data 2')
            axes[1].plot(predictions_2[i - 5:i + 5, :].ravel(), label='Predictions 2')
            axes[1].legend()
            if (whichChannelHasSeizure[1] == 1):
                axes[1].set_title("Seizure in this channel")

            axes[2].plot(data_3[i - 5:i + 5, :].ravel(), label='Data 3')
            axes[2].plot(predictions_3[i - 5:i + 5, :].ravel(), label='Predictions 3')
            axes[2].legend()
            if (whichChannelHasSeizure[2] == 1):
                axes[2].set_title("Seizure in this channel")

            axes[3].plot(data_4[i - 5:i + 5, :].ravel(), label='Data 4')
            axes[3].plot(predictions_4[i - 5:i + 5, :].ravel(), label='Predictions 4')
            axes[3].legend()
            if (whichChannelHasSeizure[3] == 1):
                axes[2].set_title("Seizure in this channel")

            plt.tight_layout()  # Adjust layout to prevent overlapping
            plt.show()

def getThresholdsPerPatient(patient_number, channel_number, sz_num):
    #sz_num = countNumberOfSeizuresPerPerson(patient_number)

    avg_list = []
    std_list = []

    for sz in range(1, sz_num + 1):
        file_path_1 = os.path.join(folder_path, f"pat_{patient_number}_sz_{sz}_ch_{channel_number}.npy")
        data_1 = np.load(file_path_1)
        mean_val_1 = np.mean(data_1)
        std_val_1 = np.std(data_1)
        new_normalized_data_1 = (data_1 - mean_val_1) / std_val_1
        new_normalized_data_1 = (new_normalized_data_1) / (np.max(new_normalized_data_1) - np.min(new_normalized_data_1))
        ll = linelength(new_normalized_data_1)

        avg = np.average(ll)
        std = np.std(ll)

        avg_list.append(avg)
        std_list.append(std)

    avg_list = np.array(avg_list)
    std_list = np.array(std_list)

    avg = np.average(avg_list)
    std = np.sqrt(np.sum(std_list ** 2)/sz_num)

    thresholds = [avg, avg + std, avg + 2 * std, avg + 3 * std, avg + 4 * std, avg + 5 * std, avg + 6 * std]

    return thresholds


def getThresholdsPerPatientAfterCleaning(path, patient_number, channel_number, sz_num):
    #sz_num = countNumberOfSeizuresPerPerson(patient_number)

    avg_list = []
    std_list = []

    for sz in range(1, sz_num + 1):

        predicted_data_1 = np.load(os.path.join(path, f"pat_{patient_number}_sz_{sz}_ch_{channel_number}.npy"))

        ll = linelength(predicted_data_1)

        avg = np.average(ll)
        std = np.std(ll)

        avg_list.append(avg)
        std_list.append(std)

    avg_list = np.array(avg_list)
    std_list = np.array(std_list)

    avg = np.average(avg_list)
    std = np.sqrt(np.sum(std_list ** 2) / sz_num)

    thresholds = [avg, avg + std, avg + 2 * std, avg + 3 * std, avg + 4 * std, avg + 5 * std, avg + 6 * std]

    return thresholds


def getImprovedResult(true_label, label_from_raw_data, predicted_label, selected_threshold):
    true_index = np.where(true_label == 1)[0]

    # predicted_label = np.all(predicted_label, axis=1)
    # label_from_raw_data = np.all(label_from_raw_data, axis=1)
    #
    # predicted_label = predicted_label.astype(int)
    # label_from_raw_data = label_from_raw_data.astype(int)

    predicted_label_1 = predicted_label[:, selected_threshold]
    label_from_raw_data_1 = label_from_raw_data[:, selected_threshold]

    predicted_index = np.where(predicted_label_1 == 1)[0]
    raw_index_negated = np.where(label_from_raw_data_1 == 0)[0]

    # intersection = np.intersect1d(true_index, predicted_index)
    # intersection = np.intersect1d(intersection, negative_label_from_raw_data_index)

    improved_label = np.zeros(len(true_label))
    for j in range(0, len(true_label)):
        if label[j] == 1 and predicted_label_1[j] == 1 and label_from_raw_data_1[j] == 0:
            improved_label[j] = 1

    intersection = np.where(improved_label == 1)[0]
    return intersection


def postProcessFP_ConsecValus(true_label, predicted_label):

    window_length = 30
    FP_list = []
    for i in range(0, predicted_label.shape[1]):
        # for line length
        conf_mat_label = []
        predicted_label_col = predicted_label[:, i]
        # for theta band power
        # predicted_label_col = predicted_label.squeeze(0)
        # predicted_label_col = predicted_label_col[:,i]
        for j in range(0, len(true_label)):
            if true_label[j] == 1 and predicted_label_col[j] == 1:
                conf_mat_label.append("TP")
            elif true_label[j] == 0 and predicted_label_col[j] == 1:
                conf_mat_label.append("FP")
            elif true_label[j] == 1 and predicted_label_col[j] == 0:
                conf_mat_label.append("FN")
            elif true_label[j] == 0 and predicted_label_col[j] == 0:
                conf_mat_label.append("TN")



        cons_counter = 0
        if(conf_mat_label[0] == "FP"):
            cons_counter = 1
        for j in range(1, len(conf_mat_label) - 1):
            if conf_mat_label[j] == "FP" and conf_mat_label[j - 1] != "FP" and conf_mat_label[j + 1] != "FP":
                cons_counter += 1

        if(conf_mat_label[len(conf_mat_label) - 2] != "FP" and conf_mat_label[len(conf_mat_label) - 1] == "FP"):
            cons_counter += 1



        #total_FP_count = sum(inner_counter)
        FP_list.append(cons_counter)

    return FP_list






def postProcessFP(true_label, predicted_label):

    window_length = 30
    FP_list = []
    for i in range(0, predicted_label.shape[1]):
        # for line length
        conf_mat_label = []
        predicted_label_col = predicted_label[:, i]
        # for theta band power
        # predicted_label_col = predicted_label.squeeze(0)
        # predicted_label_col = predicted_label_col[:,i]
        for j in range(0, len(true_label)):
            if true_label[j] == 1 and predicted_label_col[j] == 1:
                conf_mat_label.append("TP")
            elif true_label[j] == 0 and predicted_label_col[j] == 1:
                conf_mat_label.append("FP")
            elif true_label[j] == 1 and predicted_label_col[j] == 0:
                conf_mat_label.append("FN")
            elif true_label[j] == 0 and predicted_label_col[j] == 0:
                conf_mat_label.append("TN")

        counter_number = len(true_label) //window_length

        inner_counter = []
        for k in range(0, counter_number):
            epoched_conf_mat_label = conf_mat_label[k*window_length:(k+1)*window_length]
            if epoched_conf_mat_label.count("FP") >= 1:
                inner_counter.append(1)
            else:
                inner_counter.append(0)
        total_FP_count = sum(inner_counter)
        FP_list.append(total_FP_count)

    return FP_list

def concatAllResults(true_label, predicted_1, predicted_2, predicted_3, predicted_4):
    return_list = []
    for i in range(0, predicted_1.shape[1]):
        col_1 = predicted_1[:, i]
        col_1 = col_1.reshape(col_1.shape[0], 1)
        col_2 = predicted_2[:, i]
        col_2 = col_2.reshape(col_1.shape[0], 1)
        col_3 = predicted_3[:, i]
        col_3 = col_3.reshape(col_1.shape[0], 1)
        col_4 = predicted_4[:, i]
        col_4 = col_4.reshape(col_1.shape[0], 1)

        col_concat = np.concatenate([col_1, col_2, col_3, col_4], axis=1)

        return_list.append(col_concat)

    return return_list


conf_list_all = []
conf_list_actual_data_all = []
conf_list_lowpass_all = []

classified_as_sz_ch_1_actual_data = []
classified_as_sz_ch_2_actual_data = []
classified_as_sz_ch_3_actual_data = []
classified_as_sz_ch_4_actual_data = []

classified_as_sz_ch_1_predicted = []
classified_as_sz_ch_2_predicted = []
classified_as_sz_ch_3_predicted = []
classified_as_sz_ch_4_predicted = []

classified_at_least_once_new = []
classified_at_least_once_old = []
classified_at_least_once_45 = []

classified_at_least_once_10sec_new = []
classified_at_least_once_10sec_old = []
classified_at_least_once_10sec_45 = []

# eventBasedFP =[]
# eventBasedFP_old = []
# eventBasedFP_45 = []

conf_fp_new = []
conf_fp_old = []
conf_fp_45 = []

postProcessedFPList_new = []
postProcessedFPList_old = []
postProcessedFPList_45 = []

for p in range(1, 51):

    #cProfile.run("countNumberOfSeizuresPerPerson(p)")
    sz_num = countNumberOfSeizuresPerPerson(p)

    conf_list = []
    conf_list_actual_data = []
    conf_list_45 = []

    thresholds_old_ch_1 = getThresholdsPerPatient(p, 1, sz_num)
    thresholds_old_ch_2 = getThresholdsPerPatient(p, 2, sz_num)
    thresholds_old_ch_3 = getThresholdsPerPatient(p, 3, sz_num)
    thresholds_old_ch_4 = getThresholdsPerPatient(p, 4, sz_num)

    path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_norm_method"
    thresholds_new_ch_1 = getThresholdsPerPatientAfterCleaning(path, p, 1, sz_num)
    thresholds_new_ch_2 = getThresholdsPerPatientAfterCleaning(path, p, 2, sz_num)
    thresholds_new_ch_3 = getThresholdsPerPatientAfterCleaning(path, p, 3, sz_num)
    thresholds_new_ch_4 = getThresholdsPerPatientAfterCleaning(path, p, 4, sz_num)

    for sz in range(1, sz_num + 1):
        file_path_1 = os.path.join(folder_path, f"pat_{p}_sz_{sz}_ch_1.npy")
        file_path_2 = os.path.join(folder_path, f"pat_{p}_sz_{sz}_ch_2.npy")
        file_path_3 = os.path.join(folder_path, f"pat_{p}_sz_{sz}_ch_3.npy")
        file_path_4 = os.path.join(folder_path, f"pat_{p}_sz_{sz}_ch_4.npy")

        data_1 = np.load(file_path_1)
        data_2 = np.load(file_path_2)
        data_3 = np.load(file_path_3)
        data_4 = np.load(file_path_4)

        mean_val_1 = np.mean(data_1)
        std_val_1 = np.std(data_1)
        new_normalized_data_1 = (data_1 - mean_val_1) / std_val_1
        new_normalized_data_1 = (new_normalized_data_1) / (np.max(new_normalized_data_1) - np.min(new_normalized_data_1))

        mean_val_2 = np.mean(data_2)
        std_val_2 = np.std(data_2)
        new_normalized_data_2 = (data_2 - mean_val_2) / std_val_2
        new_normalized_data_2 = (new_normalized_data_2) / (np.max(new_normalized_data_2) - np.min(new_normalized_data_2))

        mean_val_3 = np.mean(data_3)
        std_val_3 = np.std(data_3)
        new_normalized_data_3 = (data_3 - mean_val_3) / std_val_3
        new_normalized_data_3 = (new_normalized_data_3) / (np.max(new_normalized_data_3) - np.min(new_normalized_data_3))

        mean_val_4 = np.mean(data_4)
        std_val_4 = np.std(data_4)
        new_normalized_data_4 = (data_4 - mean_val_4) / std_val_4
        new_normalized_data_4 = (new_normalized_data_4) / (np.max(new_normalized_data_4) - np.min(new_normalized_data_4))

        file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{sz}_labels.npy")
        label = np.load(file_path_labels)


        ####filtering with lowpass filter

        filteredSignal_1_45 = nk.signal_filter(new_normalized_data_1, sampling_rate=250, highcut=40,
                                             method='butterworth', order=4)
        filteredSignal_2_45 = nk.signal_filter(new_normalized_data_2, sampling_rate=250, highcut=40,
                                                method='butterworth', order=4)
        filteredSignal_3_45 = nk.signal_filter(new_normalized_data_3, sampling_rate=250, highcut=40,
                                                method='butterworth', order=4)
        filteredSignal_4_45 = nk.signal_filter(new_normalized_data_4, sampling_rate=250, highcut=40,
                                                method='butterworth', order=4)


        # #####when the predictions are already calculated
        predicted_data_1 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize", f"pat_{p}_sz_{sz}_ch_1.npy"))
        #predicted_data_1 = predicted_data_1.squeeze(-1)

        predicted_data_2 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize", f"pat_{p}_sz_{sz}_ch_2.npy"))
        #predicted_data_2 = predicted_data_2.squeeze(-1)

        predicted_data_3 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize", f"pat_{p}_sz_{sz}_ch_3.npy"))
        #predicted_data_3 = predicted_data_3.squeeze(-1)

        predicted_data_4 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize", f"pat_{p}_sz_{sz}_ch_4.npy"))
        #predicted_data_4 = predicted_data_4.squeeze(-1)

        #####when the predictions are not calculated yet
        # model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\lstm_encoder_bigger.h5")
        # model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
        # model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_checkPoint.h5")
        # predicted_data_1 = model.predict(new_normalized_data_1)
        # predicted_data_2 = model.predict(new_normalized_data_2)
        # predicted_data_3 = model.predict(new_normalized_data_3)
        # predicted_data_4 = model.predict(new_normalized_data_4)
        #
        # predicted_data_1 = predicted_data_1.squeeze(-1)
        # predicted_data_2 = predicted_data_2.squeeze(-1)
        # predicted_data_3 = predicted_data_3.squeeze(-1)
        # predicted_data_4 = predicted_data_4.squeeze(-1)
        #
        # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering", f"pat_{p}_sz_{sz}_ch_1.npy"), predicted_data_1)
        # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering", f"pat_{p}_sz_{sz}_ch_2.npy"), predicted_data_2)
        # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering", f"pat_{p}_sz_{sz}_ch_3.npy"), predicted_data_3)
        # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering", f"pat_{p}_sz_{sz}_ch_4.npy"), predicted_data_4)




        labels_1 = getOnlyLabels(new_normalized_data_1, label, thresholds_old_ch_1)
        labels_2 = getOnlyLabels(new_normalized_data_2, label, thresholds_old_ch_2)
        labels_3 = getOnlyLabels(new_normalized_data_3, label, thresholds_old_ch_3)
        labels_4 = getOnlyLabels(new_normalized_data_4, label, thresholds_old_ch_4)

        # label_raw_data = np.logical_or(labels_1, labels_2)
        # label_raw_data = np.logical_or(label_raw_data, labels_3)
        # label_raw_data = np.logical_or(label_raw_data, labels_4)
        # label_raw_data = label_raw_data.astype(int)
        # concat_result_raw = concatAllResults(label, labels_1, labels_2, labels_3, labels_4)
        label_raw_data = doLogicalOR(labels_1, labels_2, labels_3, labels_4)

        #innerFPWithPP_old = postProcessFP(label, label_raw_data)
        innerFPWithPP_old = postProcessFP_ConsecValus(label, label_raw_data)
        postProcessedFPList_old.append(innerFPWithPP_old)



        labels_45_1 = getOnlyLabels(filteredSignal_1_45, label, thresholds_old_ch_1)
        labels_45_2 = getOnlyLabels(filteredSignal_2_45, label, thresholds_old_ch_2)
        labels_45_3 = getOnlyLabels(filteredSignal_3_45, label,  thresholds_old_ch_3)
        labels_45_4 = getOnlyLabels(filteredSignal_4_45, label, thresholds_old_ch_4)
        labels_45_data = doLogicalOR(labels_45_1, labels_45_2, labels_45_3, labels_45_4)

        # labels_45_data = np.logical_or(labels_45_1, labels_45_2)
        # labels_45_data = np.logical_or(labels_45_data, labels_45_3)
        # labels_45_data = np.logical_or(labels_45_data, labels_45_4)
        # labels_45_data = labels_45_data.astype(int)

        #innerFPWithPP_45 = postProcessFP(label, labels_45_data)
        innerFPWithPP_45 = postProcessFP_ConsecValus(label, labels_45_data)
        postProcessedFPList_45.append(innerFPWithPP_45)




        classified_at_least_once_old.append(classifiedAtLeastOnce(label, label_raw_data))
        classified_at_least_once_10sec_old.append(classifiedAtLeastOnce_10sec(label, label_raw_data))


        classified_at_least_once_45.append(classifiedAtLeastOnce(label, labels_45_data))
        classified_at_least_once_10sec_45.append(classifiedAtLeastOnce_10sec(label, labels_45_data))

        #print(getNoneDetections(label, labels_1, labels_2, labels_3, labels_4))
        #false_detections_raw = getFlaseDetections(label, label_raw_data)

        #conf_mat_raw_data = confusion_matrix(label, label_raw_data)
        conf_mat_raw_data = []
        for i in range(0, label_raw_data.shape[1]):

            #for line length
            label_col = label_raw_data[:, i]
            #for theta band power
            # label_col = label_raw_data.squeeze(0)
            # label_col = label_col[:, i]
            conf = confusion_matrix(label, label_col).ravel()
            conf_mat_raw_data.append(conf)

        conf_mat_45 = []
        for i in range(0, label_raw_data.shape[1]):
            # for line length
            label_col = label_raw_data[:, i]
            # for theta band power
            #label_col = labels_45_data.squeeze(0)
            #label_col = label_col[:, i]

            conf = confusion_matrix(label, label_col).ravel()
            conf_mat_45.append(conf)


        predicted_labels_1 = getOnlyLabels(predicted_data_1, label, thresholds_new_ch_1)
        predicted_labels_2 = getOnlyLabels(predicted_data_2, label, thresholds_new_ch_2)
        predicted_labels_3 = getOnlyLabels(predicted_data_3, label, thresholds_new_ch_3)
        predicted_labels_4 = getOnlyLabels(predicted_data_4, label, thresholds_new_ch_4)

        if (np.array_equal(predicted_labels_1, predicted_labels_2) or np.array_equal(predicted_labels_2,
                                                                                      predicted_labels_3) or np.array_equal(
                predicted_labels_3, predicted_labels_4)):
            print("all equal")


        predicted_label = doLogicalOR(predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4)
        # predicted_label = np.logical_or(predicted_labels_1, predicted_labels_2)
        # predicted_label = np.logical_or(predicted_label, predicted_labels_3)
        # predicted_label = np.logical_or(predicted_label, predicted_labels_4)
        # predicted_label = predicted_label.astype(int)


        classified_at_least_once_new.append(classifiedAtLeastOnce(label, predicted_label))
        classified_at_least_once_10sec_new.append(classifiedAtLeastOnce_10sec(label, predicted_label))

        #innerFPWithPP_new = postProcessFP(label, predicted_label)
        innerFPWithPP_new = postProcessFP_ConsecValus(label, predicted_label)
        postProcessedFPList_new.append(innerFPWithPP_new)


        conf_mat = []
        #conf_mat = confusion_matrix(label, predicted_label)
        for i in range(0, predicted_label.shape[1]):
            #for line length
            predicted_label_col = predicted_label[:,i]
            #for theta band power
            # predicted_label_col = predicted_label.squeeze(0)
            # predicted_label_col = predicted_label_col[:,i]
            conf = confusion_matrix(label, predicted_label_col).ravel()
            conf_mat.append(conf)

        selected_threshold_list = getTheBestThresholds(label, predicted_label)

        if(len(selected_threshold_list) != 0):
            selected_threshold = selected_threshold_list[-1]
            #print(selected_threshold)

            if(selected_threshold != 0 and selected_threshold != 1):

                #false_detections_predicted = getFlaseDetections(label, predicted_label, label_raw_data, selected_threshold)
                false_detections_predicted = getReducedFlaseDetections(label, predicted_label, label_raw_data, selected_threshold)
                plotFalsePositives(false_detections_predicted, new_normalized_data_1, new_normalized_data_2, new_normalized_data_3, new_normalized_data_4, predicted_data_1, predicted_data_2, predicted_data_3, predicted_data_4, p, sz, selected_threshold, predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4)

                improved_result = getImprovedResult(label, label_raw_data, predicted_label, selected_threshold)
                #plotFalsePositives(improved_result, new_normalized_data_1, new_normalized_data_2, new_normalized_data_3, new_normalized_data_4, predicted_data_1, predicted_data_2, predicted_data_3, predicted_data_4, p, sz, selected_threshold, predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4)

            dismproved_result = getdisImprovement(label, label_raw_data, predicted_label, selected_threshold)
            #plotFalsePositives(dismproved_result, new_normalized_data_1, new_normalized_data_2, new_normalized_data_3,new_normalized_data_4, predicted_data_1, predicted_data_2, predicted_data_3,predicted_data_4, p, sz, selected_threshold, labels_1, labels_2, labels_3, labels_4)



        conf_list.append(conf_mat)
        conf_list_actual_data.append(conf_mat_raw_data)
        conf_list_45.append(conf_mat_45)


    conf_list_all.append(conf_list)
    conf_list_actual_data_all.append(conf_list_actual_data)
    conf_list_lowpass_all.append(conf_list_45)

    #print(len(conf_list_all))

averaged_list = []
averaged_list_actual_data = []
averaged_list_45 = []

ppofFP_new_array = np.array(postProcessedFPList_new)
ppofFP_new_array = np.mean(ppofFP_new_array, axis=0)

ppofFP_old_array = np.array(postProcessedFPList_old)
ppofFP_old_array = np.mean(ppofFP_old_array, axis=0)

ppofFP_45_array = np.array(postProcessedFPList_45)
ppofFP_45_array = np.mean(ppofFP_45_array, axis=0)

for patient in conf_list_all:
    patient_array = np.array(patient)
    patient_array = np.mean(patient_array, axis=0)
    averaged_list.append(patient_array)

for patient in conf_list_actual_data_all:
    patient_array = np.array(patient)
    patient_array = np.mean(patient_array, axis=0)
    averaged_list_actual_data.append(patient_array)

for patient in conf_list_lowpass_all:
    patient_array = np.array(patient)
    patient_array = np.mean(patient_array, axis=0)
    averaged_list_45.append(patient_array)



averaged_list = np.array(averaged_list)
averaged_list_actual_data = np.array(averaged_list_actual_data)
averaged_list_45 = np.array(averaged_list_45)

averaged_list = np.mean(averaged_list, axis=0)
averaged_list_actual_data = np.mean(averaged_list_actual_data, axis=0)
averaged_list_45 = np.mean(averaged_list_45, axis=0)

classified_at_least_once_new = np.array(classified_at_least_once_new)
classified_at_least_once_old = np.array(classified_at_least_once_old)
classified_at_least_once_45 = np.array(classified_at_least_once_45)


#classified_at_least_once_new = np.mean(classified_at_least_once_new, axis=0)
classified_at_least_once_new = np.sum(classified_at_least_once_new, axis=0)/362
#classified_at_least_once_old = np.mean(classified_at_least_once_old, axis=0)
classified_at_least_once_old = np.sum(classified_at_least_once_old, axis=0)/362
classified_at_least_once_45 = np.mean(classified_at_least_once_45, axis=0)

classified_at_least_once_10sec_new = np.array(classified_at_least_once_10sec_new)
classified_at_least_once_10sec_old = np.array(classified_at_least_once_10sec_old)
classified_at_least_once_10sec_45 = np.array(classified_at_least_once_10sec_45)

#classified_at_least_once_10sec_new = np.mean(classified_at_least_once_10sec_new, axis=0)
classified_at_least_once_10sec_new = np.sum(classified_at_least_once_10sec_new, axis=0)/362
#classified_at_least_once_10sec_old = np.mean(classified_at_least_once_10sec_old, axis=0)
classified_at_least_once_10sec_old = np.sum(classified_at_least_once_10sec_old, axis=0)/362
classified_at_least_once_10sec_45 = np.mean(classified_at_least_once_10sec_45, axis=0)


# plt.plot(averaged_list[:, 1], classified_at_least_once_new)
# plt.plot(averaged_list_actual_data[:, 1], classified_at_least_once_old)
# plt.plot(averaged_list_45[:, 1], classified_at_least_once_45)
# auc_actual = auc(averaged_list_actual_data[:, 1], classified_at_least_once_new)
# auc_predicted = auc(averaged_list[:, 1], classified_at_least_once_old)
# auc_45 = auc(averaged_list_45[:, 1], classified_at_least_once_45)
# plt.xlabel("False Positive Rate")
# plt.ylabel("classified at least once as seizure")
# legend_labels = ["new (AUC={:.2f})".format(auc_actual),
#                  "old (AUC={:.2f})".format(auc_predicted),
#                  "45 (AUC={:.2f})".format(auc_45)]
#
# plt.legend(legend_labels)
# plt.show()





plt.plot(np.sort(ppofFP_new_array)[::-1], classified_at_least_once_new)
plt.plot(ppofFP_old_array, classified_at_least_once_old)
plt.plot(ppofFP_45_array, classified_at_least_once_45)
auc_actual = auc(np.sort(ppofFP_new_array)[::-1], classified_at_least_once_new)
auc_predicted = auc(ppofFP_old_array, classified_at_least_once_old)
auc_45 = auc(ppofFP_45_array, classified_at_least_once_45)
plt.xlabel("False Positive Rate")
plt.ylabel("classified at least once as seizure")
legend_labels = ["new (AUC={:.2f})".format(auc_actual),
                 "old (AUC={:.2f})".format(auc_predicted),
                 "45 (AUC={:.2f})".format(auc_45)]

plt.legend(legend_labels)
plt.show()
#
# plt.plot(ppofFP_new_array, classified_at_least_once_10sec_new)
# plt.plot(ppofFP_old_array, classified_at_least_once_10sec_old)
# plt.plot(ppofFP_45_array, classified_at_least_once_10sec_45)
# auc_actual = auc(ppofFP_new_array, classified_at_least_once_10sec_new)
# auc_predicted = auc(ppofFP_old_array, classified_at_least_once_10sec_old)
# auc_45 = auc(ppofFP_45_array, classified_at_least_once_10sec_45)
# plt.xlabel("False Positive Rate")
# plt.ylabel("classified at least once as seizure")
# legend_labels = ["new (AUC={:.2f})".format(auc_actual),
#                  "old (AUC={:.2f})".format(auc_predicted),
#                  "45 (AUC={:.2f})".format(auc_45)]

plt.legend(legend_labels)
plt.show()

# plt.plot(averaged_list[:, 1], classified_at_least_once_10sec_new)
# plt.plot(averaged_list_actual_data[:, 1], classified_at_least_once_10sec_old)
# plt.plot(averaged_list_45[:, 1], classified_at_least_once_10sec_45)
# auc_actual = auc(averaged_list_actual_data[:, 1], classified_at_least_once_10sec_new)
# auc_predicted = auc(averaged_list[:, 1], classified_at_least_once_10sec_old)
# auc_45 = auc(averaged_list_45[:, 1], classified_at_least_once_10sec_45)
# plt.xlabel("False Positive Rate")
# plt.ylabel("classified at least once as seizure in the first 10 seconds")
# legend_labels = ["new (AUC={:.2f})".format(auc_actual),
#                     "old (AUC={:.2f})".format(auc_predicted),
#                     "45 (AUC={:.2f})".format(auc_45)]
# plt.legend(legend_labels)
# plt.show()


# plt.plot(averaged_list[:, 1], averaged_list[:, 3])
# plt.plot(averaged_list_actual_data[:, 1], averaged_list_actual_data[:, 3])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
#
# plt.legend(["predicted", "actual data"])
# plt.show()
#

# # Iterate through each element in the original list
# for element in conf_list_all:
#     # Convert the element to a NumPy array
#     element_array = np.array(element)
#
#     # Calculate the average along axis 0
#     averaged_array = np.mean(element_array, axis=0)
#
#     # Append the result to the new list
#     averaged_list.append(averaged_array)
#
#
#
# for element in conf_list_actual_data_all:
#     # Convert the element to a NumPy array
#     element_array = np.array(element)
#
#     # Calculate the average along axis 0
#     averaged_array = np.mean(element_array, axis=0)
#
#     # Append the result to the new list
#     averaged_list_actual_data.append(averaged_array)
#
#
#
#
#
#
# averaged_list = np.array(averaged_list)
# averaged_list = np.mean(averaged_list, axis=0)
#
# averaged_list_actual_data = np.array(averaged_list_actual_data)
# averaged_list_actual_data = np.mean(averaged_list_actual_data, axis=0)
#
#
# classified_as_sz_ch_1_actual_data = np.array(classified_as_sz_ch_1_actual_data)
# classified_as_sz_ch_2_actual_data = np.array(classified_as_sz_ch_2_actual_data)
# classified_as_sz_ch_3_actual_data = np.array(classified_as_sz_ch_3_actual_data)
# classified_as_sz_ch_4_actual_data = np.array(classified_as_sz_ch_4_actual_data)
#
# res_actual = np.logical_or(classified_as_sz_ch_1_actual_data, classified_as_sz_ch_2_actual_data)
# res_actual = np.logical_or(res_actual, classified_as_sz_ch_3_actual_data)
# res_actual = np.logical_or(res_actual, classified_as_sz_ch_4_actual_data)
#
# classified_as_sz_ch_1_predicted = np.array(classified_as_sz_ch_1_predicted)
# classified_as_sz_ch_2_predicted = np.array(classified_as_sz_ch_2_predicted)
# classified_as_sz_ch_3_predicted = np.array(classified_as_sz_ch_3_predicted)
# classified_as_sz_ch_4_predicted = np.array(classified_as_sz_ch_4_predicted)
#
# res_predicted = np.logical_or(classified_as_sz_ch_1_predicted, classified_as_sz_ch_2_predicted)
# res_predicted = np.logical_or(res_predicted, classified_as_sz_ch_3_predicted)
# res_predicted = np.logical_or(res_predicted, classified_as_sz_ch_4_predicted)
#
#
# average_classified_as_sz_actual_data = np.mean(res_actual, axis=0)
# average_classified_as_sz_predicted = np.mean(res_predicted, axis=0)
#
#
#
#
# plt.plot(averaged_list[:, 1], averaged_list[:, 3])
# plt.plot(averaged_list_actual_data[:, 1], averaged_list_actual_data[:, 3])
# plt.legend(["predicted", "actual data"])
# plt.show()
#
# auc_actual = auc(averaged_list_actual_data[:, 1]/362, average_classified_as_sz_actual_data)
# auc_predicted = auc(averaged_list[:, 1]/362, average_classified_as_sz_predicted)
#
# print("auc actual data: ", auc_actual)
# print("auc predicted: ", auc_predicted)
#
# plt.plot(averaged_list_actual_data[:, 1]/362, average_classified_as_sz_actual_data)
# plt.plot(averaged_list[:, 1]/362, average_classified_as_sz_predicted)
# plt.legend(["actual data", "predicted"])
# plt.xlabel('FP/hour')
# plt.ylabel('Classifies as seizure at least once')
# plt.show()

#
#
# auc_list_new = []
# auc_list_old = []
# auc_list_bandpass = []
# auc_list_30 = []
# auc_list_45 = []
# auc_list_70 = []
#
# TP_list_new = []
# FP_list_new = []
#
# TP_list_old = []
# FP_list_old = []
#
# TP_list_lowpass = []
# FP_list_lowpass = []
#
# TN_list_new = []
# FN_list_new = []
#
# TN_list_old = []
# FN_list_old = []
#
# cm_45_list = []
# cm_raw_list = []
# cm_predicted_list = []
#
# classified_as_sz_list_45 = []
# classified_as_sz_list_raw = []
# classified_as_sz_list_predicted = []
#
#
# #model_2 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_more_filters.h5')
# #model_2 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN_bigger_kernel.h5')
# #model_2 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_checkpoint.h5')
#
#
# no_p_no_sz = 362
# all_channels_output_new = []
# all_channels_output_old = []
#
# for p in range(10,11):
#     sz_num = countNumberOfSeizuresPerPerson(p)
#     #no_p_no_sz += sz_num
#     for i in range(1, sz_num + 1):
#         all_ch_labels_new = []
#         all_ch_labels_old = []
#         data_appended = []
#         for ch in range(1, 5):
#             file_path = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_{ch}.npy")
#             data = np.load(file_path)
#             file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{i}_labels.npy")
#
#             label = np.load(file_path_labels)
#
#             # file_path_1 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_1.npy")
#             # file_path_2 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_2.npy")
#             # file_path_3 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_3.npy")
#             # file_path_4 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_4.npy")
#             #
#             # data_1 = np.load(file_path_1)
#             # data_2 = np.load(file_path_2)
#             # data_3 = np.load(file_path_3)
#             # data_4 = np.load(file_path_4)
#             #
#             # data_1, data_2, data_3, data_4 = normalize_ch_data(data_1, data_2, data_3, data_4)
#             #
#             # data = np.empty_like(data_1)
#             # if (ch == 1):
#             #     data = data_1
#             # elif (ch == 2):
#             #     data = data_2
#             # elif (ch == 3):
#             #     data = data_3
#             # elif (ch == 4):
#             #     data = data_4
#
#             mean_val = np.mean(data)
#             std_val = np.std(data)
#             # Normalize the data to the range [-1, 1]
#             new_normalized_data = (data - mean_val) / std_val
#             new_normalized_data = (new_normalized_data) / (np.max(new_normalized_data) - np.min(new_normalized_data))
#
#             data_appended.append(new_normalized_data)
#
#             #num_zeros = (0, 12)
#
#             # Pad the array with zeros
#             #padded_data = np.pad(data, ((0, 0), num_zeros), mode='constant')
#
#             # predicted_data = model_2.predict(new_normalized_data)
#             # predicted_data = predicted_data.squeeze(-1)
#             # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_norm_method", f"pat_{p}_sz_{i}_ch_{ch}.npy"), predicted_data)
#             #np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\CNN\result_cnn", f"pat_{p}_sz_{i}_ch_{ch}.npy"), predicted_data)
#             #np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\GRU_cheby", f"pat_{p}_sz_{i}_ch_{ch}.npy"), predicted_data)
#             #predicted_data = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\CNN\result_cnn", f"pat_{p}_sz_{i}_ch_{ch}.npy"))
#             #predicted_data = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby", f"pat_{p}_sz_{i}_ch_{ch}.npy"))
#             predicted_data = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_norm_method", f"pat_{p}_sz_{i}_ch_{ch}.npy"))
#             #predicted_data = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\GRU_cheby", f"pat_{p}_sz_{i}_ch_{ch}.npy"))
#             # predicted_data = model_2.predict(padded_data)
#             # predicted_data = predicted_data.squeeze(-1)
#             #predicted_data = sharpenSignal(predicted_data)
#
#             filteredSignal_45 = nk.signal_filter(data, sampling_rate=250, highcut=40,
#                                                method='butterworth', order=4)
#             # filteredSignal_70 = nk.signal_filter(data, sampling_rate=250, lowcut=0.1, highcut=70,
#             #                                      method='butterworth', order=4)
#             # filteredSignal_30 = nk.signal_filter(data, sampling_rate=250, lowcut=0.1, highcut=30,
#             #                                      method='butterworth', order=4)
#
#
#             new_ll = linelength(predicted_data)
#             old_ll = linelength(data)
#             # ll_30 = linelength(filteredSignal_30)
#             ll_45 = linelength(filteredSignal_45)
#             # ll_70 = linelength(filteredSignal_70)
#
#             # new_ll = thetaBandPower(predicted_data)
#             # old_ll = thetaBandPower(data)
#             # # ll_30 = thetaBandPower(filteredSignal_30)
#             # ll_45 = thetaBandPower(filteredSignal_45)
#             # ll_70 = thetaBandPower(filteredSignal_70)
#
#
#
#             # avg_30 = np.average(ll_30)
#             # std_30 = np.std(ll_30)
#             # thresholds_30 = [avg_30 - 3 * std_30, avg_30 - 2 * std_30, avg_30 - std_30, avg_30, avg_30 + std_30, avg_30 + 2 * std_30, avg_30 + 3 * std_30]
#             #
#             #
#             # sens_30 = []
#             # spec_30 = []
#             # for th in thresholds_30:
#             #     new_ll_label = (ll_30 > th).astype(int)
#             #     sens1 = sensitivity(new_ll_label, label)
#             #     spec1 = specificity(new_ll_label, label)
#             #     sens_30.append(sens1["sens"])
#             #     spec_30.append(1 - spec1["spec"])
#
#             avg_45 = np.average(ll_45)
#             std_45 = np.std(ll_45)
#             thresholds_45 = [avg_45 - 3 * std_45, avg_45 - 2 * std_45, avg_45 - std_45, avg_45, avg_45 + std_45, avg_45 + 2 * std_45, avg_45 + 3 * std_45]
#
#             sens_45 = []
#             spec_45 = []
#             cm_45 = []  # confusion matrix
#             classification_list_45 = []
#             for th in thresholds_45:
#                 tn, fp, fn, tp = confusion_matrix(label, (ll_45 > th).astype(int)).ravel()
#                 cm_45.append([tn, fp, fn, tp])
#                 new_ll_label = (ll_45 > th).astype(int)
#                 classification_list_45.append(classifiedAtLeastOnce(label, new_ll_label))
#                 sens1 = sensitivity(new_ll_label, label)
#                 spec1 = specificity(new_ll_label, label)
#                 sens_45.append(sens1["TP"])
#                 spec_45.append(spec1["FP"])
#
#             TP_list_lowpass.append(mean(sens_45))
#             FP_list_lowpass.append(mean(spec_45))
#             cm_45_list.append(cm_45)
#             classified_as_sz_list_45.append(classification_list_45)
#
#             # avg_70 = np.average(ll_70)
#             # std_70 = np.std(ll_70)
#             # thresholds_70 = [avg_70 - 3 * std_70, avg_70 - 2 * std_70, avg_70 - std_70, avg_70, avg_70 + std_70, avg_70 + 2 * std_70, avg_70 + 3 * std_70]
#             #
#             # sens_70 = []
#             # spec_70 = []
#             # for th in thresholds_70:
#             #     new_ll_label = (ll_70 > th).astype(int)
#             #     sens1 = sensitivity(new_ll_label, label)
#             #     spec1 = specificity(new_ll_label, label)
#             #     sens_70.append(sens1["sens"])
#             #     spec_70.append(1 - spec1["spec"])
#             #
#             #
#
#             old_avg = np.average(old_ll)
#             old_std = np.std(old_ll)
#             thresholds_old = [old_avg - 3 * old_std, old_avg - 2 * old_std, old_avg - old_std, old_avg,
#                               old_avg + old_std, old_avg + 2 * old_std, old_avg + 3 * old_std]
#
#             avg = np.average(new_ll)
#             std = np.std(new_ll)
#             thresholds = [avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std]
#
#             sens_new = []
#             spec_new = []
#             cm_new = []  # confusion matrix
#             classification_list_new = []
#             for th in thresholds:
#                 tn, fp, fn, tp = confusion_matrix(label, (new_ll > th).astype(int)).ravel()
#                 cm_new.append([tn, fp, fn, tp])
#                 new_ll_label = (new_ll > th).astype(int)
#                 classification_list_new.append(classifiedAtLeastOnce(label, new_ll_label))
#                 sens1 = sensitivity(new_ll_label, label)
#                 spec1 = specificity(new_ll_label, label)
#                 sens_new.append(sens1["TP"])
#                 spec_new.append(spec1["FP"])
#                 all_ch_labels_new.append(new_ll_label)
#
#             TP_list_new.append(mean(sens_new))
#             FP_list_new.append(mean(spec_new))
#             cm_predicted_list.append(cm_new)
#             classified_as_sz_list_predicted.append(classification_list_new)
#
#             sens_old = []
#             spec_old = []
#             cm_old = []  # confusion matrix
#             classification_list_raw = []
#             for th in thresholds_old:
#                 tn, fp, fn, tp = confusion_matrix(label, (old_ll > th).astype(int)).ravel()
#                 cm_old.append([tn, fp, fn, tp])
#                 old_ll_label = (old_ll > th).astype(int)
#                 classification_list_raw.append(classifiedAtLeastOnce(label, old_ll_label))
#                 sens2 = sensitivity(old_ll_label, label)
#                 spec2 = specificity(old_ll_label, label)
#                 sens_old.append(sens2["TP"])
#                 spec_old.append(spec2["FP"])
#                 all_ch_labels_old.append(old_ll_label)
#
#             TP_list_old.append(mean(sens_old))
#             FP_list_old.append(mean(spec_old))
#             cm_raw_list.append(cm_old)
#             classified_as_sz_list_raw.append(classification_list_raw)
#             all_channels_output_new.append(all_ch_labels_new)
#             all_channels_output_old.append(all_ch_labels_old)
#
#             # print("false positive")
#             # print(spec_new)
#             # print("average FP new", mean(spec_new))
#             # print(spec_old)
#             # print("average FP old", mean(spec_old))
#             #
#             #
#             # print("true positive")
#             # print(sens_new)
#             # print("average TP new", mean(sens_new))
#             # print(sens_old)
#             # print("average TP old", mean(sens_old))
#
#
#
# #             auc_new = auc(spec_new, sens_new)
# #             auc_old = auc(spec_old, sens_old)
# #             # auc_30 = auc(spec_30, sens_30)
# #             #auc_45 = auc(spec_45, sens_45)
# #             # auc_70 = auc(spec_70, sens_70)
# #
# #             auc_list_new.append(auc_new)
# #             auc_list_old.append(auc_old)
# #             # auc_list_30.append(auc_30)
# #             #auc_list_45.append(auc_45)
# #             # auc_list_70.append(auc_70)
# #
# #
# #             print(f"patient {p}, seizure {i}, channel {ch}, auc_new {auc_new}, auc_old {auc_old}")
# #
# #
# # print(f"mean auc_new {np.mean(auc_list_new)}, mean auc_old {np.mean(auc_list_old)}")
# #print(f"std auc_new {np.std(auc_list_new)}, std auc_old {np.std(auc_list_old)}")
# #print(f"mean auc_30 {np.mean(auc_list_30)}, mean auc_45 {np.mean(auc_list_45)}, mean auc_70 {np.mean(auc_list_70)}")
#
# ch1_list_new = all_channels_output_new[0::4]
# ch2_list_new = all_channels_output_new[1::4]
# ch3_list_new = all_channels_output_new[2::4]
# ch4_list_new = all_channels_output_new[3::4]
#
# ch1_list_old = all_channels_output_old[0::4]
# ch2_list_old = all_channels_output_old[1::4]
# ch3_list_old = all_channels_output_old[2::4]
# ch4_list_old = all_channels_output_old[3::4]
#
# for i in range(0,10):
#     if np.equal(ch1_list_new[i], ch2_list_new[i]).all():
#         print("ch1 and 2 is equal")   # all channels are equal
#
#     if np.equal(ch2_list_new[i], ch3_list_new[i]).all():
#         print("ch2 and 3 is equal")
#
#     if np.equal(ch3_list_new[i], ch4_list_new[i]).all():
#         print("ch3 and 4 is equal")
#
#
#     if np.equal(ch1_list_old[i], ch2_list_old[i]).all():
#         print("ch1 and 2 is equal old")  # all channels are equal
#
#     if np.equal(ch2_list_old[i], ch3_list_old[i]).all():
#         print("ch2 and 3 is equal old")
#
#     if np.equal(ch3_list_old[i], ch4_list_old[i]).all():
#         print("ch3 and 4 is equal old")


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


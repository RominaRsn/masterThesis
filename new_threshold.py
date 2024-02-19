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

def doTwoAndOneOr(predicted_label_1, predicted_labe_2, predicted_label_3, predicted_label_4):
    result = np.empty_like(predicted_label_1)
    for i in range(0, predicted_label_1.shape[1]):
        col_predic_1 = predicted_label_1[:, i]
        col_predic_2 = predicted_labe_2[:, i]
        col_predic_3 = predicted_label_3[:, i]
        col_predic_4 = predicted_label_4[:, i]
        res1 = np.logical_or(col_predic_1, col_predic_4)
        res2 = np.logical_or(col_predic_2, col_predic_3)
        res = np.logical_and(res1, res2)
        result[:, i] = res
    return result

def doAnds(predicted_label_1, predicted_labe_2, predicted_label_3, predicted_label_4):
    result = np.empty_like(predicted_label_1)
    for i in range(0, predicted_label_1.shape[1]):
        col_predic_1 = predicted_label_1[:, i]
        col_predic_2 = predicted_labe_2[:, i]
        col_predic_3 = predicted_label_3[:, i]
        col_predic_4 = predicted_label_4[:, i]
        res = np.logical_and(col_predic_1, col_predic_2)
        res = np.logical_and(res, col_predic_3)
        res = np.logical_and(res, col_predic_4)
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
    # ll = thetaBandPower(data)
    # ll = ll.squeeze(-1)
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
        print("Patient: ", patient_number, "Seizure: ", seizure_number, "selected threshold: ", selected_threshold)

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

def plotFalsePositives_new(index_list, data_1, data_2, data_3, data_4, predictions_1, predictions_2, predictions_3, predictions_4, patient_number, seizure_number, selected_threshold, predicted_label_1, predicted_label_2, predicted_label_3, predicted_label_4,
                           raw_labels_1, raw_labels_2, raw_labels_3, raw_labels_4):

    predicted_label_col_1 = raw_labels_1[:, selected_threshold]
    predicted_label_col_2 = raw_labels_2[:, selected_threshold]
    predicted_label_col_3 = raw_labels_3[:, selected_threshold]
    predicted_label_col_4 = raw_labels_4[:, selected_threshold]

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


            display_span = 3
            # Create subplots with specified axes
            fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharey='col')
            fig.suptitle(f'patient: {patient_number}, seizure: {seizure_number}, index: {i}')
            # Plot each subplot
            axes[0].plot(data_1[i - display_span:i + display_span, :].ravel(), label='Data 1')
            axes[0].plot(predictions_1[i - display_span:i + display_span, :].ravel(), label='Predictions 1')
            axes[0].legend()
            if(whichChannelHasSeizure[0] == 1):
                axes[0].set_title("Seizure in this channel")

            axes[1].plot(data_2[i - display_span:i + display_span, :].ravel(), label='Data 2')
            axes[1].plot(predictions_2[i - display_span:i + display_span, :].ravel(), label='Predictions 2')
            axes[1].legend()
            if (whichChannelHasSeizure[1] == 1):
                axes[1].set_title("Seizure in this channel")

            axes[2].plot(data_3[i - display_span:i + display_span, :].ravel(), label='Data 3')
            axes[2].plot(predictions_3[i - display_span:i + display_span, :].ravel(), label='Predictions 3')
            axes[2].legend()
            if (whichChannelHasSeizure[2] == 1):
                axes[2].set_title("Seizure in this channel")

            axes[3].plot(data_4[i - display_span:i + display_span, :].ravel(), label='Data 4')
            axes[3].plot(predictions_4[i - display_span:i + display_span, :].ravel(), label='Predictions 4')
            axes[3].legend()
            if (whichChannelHasSeizure[3] == 1):
                axes[2].set_title("Seizure in this channel")

            plt.tight_layout()  # Adjust layout to prevent overlapping
            plt.show()


def movingAverage(ll):
    # Define the window size for the moving average
    window_size = 10

    # Create a moving average window as a simple boxcar window
    window = np.ones(window_size) / window_size

    # Apply the moving average using convolution
    moving_avg = np.convolve(ll.ravel(), window, mode='valid')

    return moving_avg


def getThresholdsPerPatient(patient_number, channel_number, sz_num):
    #sz_num = countNumberOfSeizuresPerPerson(patient_number)

    avg_list = []
    std_list = []
    max_list = []

    for sz in range(1, sz_num + 1):
        file_path_1 = os.path.join(folder_path, f"pat_{patient_number}_sz_{sz}_ch_{channel_number}.npy")
        data_1 = np.load(file_path_1)
        mean_val_1 = np.mean(data_1)
        std_val_1 = np.std(data_1)
        new_normalized_data_1 = (data_1 - mean_val_1) / std_val_1
        new_normalized_data_1 = (new_normalized_data_1) / (np.max(new_normalized_data_1) - np.min(new_normalized_data_1))
        ll = linelength(new_normalized_data_1)


        moving_avg = movingAverage(ll)
        # ll = thetaBandPower(new_normalized_data_1)
        # ll = ll.squeeze(-1)

        avg = np.average(ll)
        std = np.std(ll)

        avg_list.append(avg)
        std_list.append(std)

        max_list.append(np.max(moving_avg))

    avg_list = np.array(avg_list)
    std_list = np.array(std_list)

    avg = np.average(avg_list)
    std = np.sqrt(np.sum(std_list ** 2)/sz_num)

    max_ll = max(max_list)

    #thresholds = [avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std, avg + 4 * std, avg + 5 * std, avg + 6 * std]
    thresholds = [max_ll * 0.5, max_ll * 0.55, max_ll * 0.6, max_ll * 0.65, max_ll * 0.7, max_ll * 0.75, max_ll * 0.8, max_ll * 0.85, max_ll * 0.9, max_ll * 0.95, max_ll * 0.99]
    return thresholds


def getThresholdsPerPatient_newThresholdMethod(patient_number, channel_number, sz_num):
    #sz_num = countNumberOfSeizuresPerPerson(patient_number)

    avg_list = []
    std_list = []

    ll_list = []
    label_list = []

    for sz in range(1, sz_num + 1):
        file_path_1 = os.path.join(folder_path, f"pat_{patient_number}_sz_{sz}_ch_{channel_number}.npy")
        data_1 = np.load(file_path_1)
        mean_val_1 = np.mean(data_1)
        std_val_1 = np.std(data_1)
        new_normalized_data_1 = (data_1 - mean_val_1) / std_val_1
        new_normalized_data_1 = (new_normalized_data_1) / (np.max(new_normalized_data_1) - np.min(new_normalized_data_1))
        ll = linelength(new_normalized_data_1)
        path_extension_labels = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s"
        file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{sz}_labels.npy")
        labels = np.load(file_path_labels)

        ll_list.append(ll)
        label_list.append(labels)

        # ll = thetaBandPower(predicted_data_1)
        # ll = ll.squeeze(-1)

    concatenated_ll = np.concatenate(ll_list)
    concatenated_labels = np.concatenate(label_list)


    fpr, tpr, thresholds = roc_curve(concatenated_labels, concatenated_ll)

    desired_tpr_values = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    # Initialize a dictionary to store thresholds corresponding to each desired TPR value
    thresholds_at_desired_tpr = []

    # Find thresholds for each desired TPR value
    for desired_tpr in desired_tpr_values:
        index = np.argmax(tpr >= desired_tpr)
        thresholds_at_desired_tpr.append(thresholds[index])

    return thresholds_at_desired_tpr




def plotLineLengthValues(index_list, data_1, data_2, data_3, data_4, data_predicted_1, data_predicted_2, data_predicted_3, data_predicted_4, selected_threshold, selected_threshold_predicted, predicted_label_1, predicted_label_2, predicted_label_3, predicted_label_4, patient_number, seizure_number):
    predicted_label_col_1 = predicted_label_1[:, selected_threshold]
    predicted_label_col_2 = predicted_label_2[:, selected_threshold]
    predicted_label_col_3 = predicted_label_3[:, selected_threshold]
    predicted_label_col_4 = predicted_label_4[:, selected_threshold]

    predicted_ones_1 = np.where(predicted_label_col_1 == 1)[0]
    predicted_ones_2 = np.where(predicted_label_col_2 == 1)[0]
    predicted_ones_3 = np.where(predicted_label_col_3 == 1)[0]
    predicted_ones_4 = np.where(predicted_label_col_4 == 1)[0]

    ll_raw_1 = linelength(data_1)
    ll_raw_2 = linelength(data_2)
    ll_raw_3 = linelength(data_3)
    ll_raw_4 = linelength(data_4)

    ll_predicted_1 = linelength(data_predicted_1)
    ll_predicted_2 = linelength(data_predicted_2)
    ll_predicted_3 = linelength(data_predicted_3)
    ll_predicted_4 = linelength(data_predicted_4)


    for i in index_list:
        print("Patient: ", patient_number, "Seizure: ", seizure_number, "selected threshold: ", selected_threshold)

        if (i > 10):
            whichChannelHasSeizure = [0, 0, 0, 0]
            if (i in predicted_ones_1):
                whichChannelHasSeizure[0] = 1
            if (i in predicted_ones_2):
                whichChannelHasSeizure[1] = 1
            if (i in predicted_ones_3):
                whichChannelHasSeizure[2] = 1
            if (i in predicted_ones_4):
                whichChannelHasSeizure[3] = 1
            for k in range(4):
                if(whichChannelHasSeizure[k] == 1):
                    ll_raw = []
                    ll_predicted = []
                    if (k == 0):
                        ll_raw = ll_raw_1
                        ll_predicted = ll_predicted_1
                    elif (k == 1):
                        ll_raw = ll_raw_2
                        ll_predicted = ll_predicted_2
                    elif (k == 2):
                        ll_raw = ll_raw_3
                        ll_predicted = ll_predicted_3
                    elif (k == 3):
                        ll_raw = ll_raw_4
                        ll_predicted = ll_predicted_4

                # Create subplots with specified axes
                    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharey='col')
                    fig.suptitle(f'patient: {patient_number}, seizure: {seizure_number}, index: {i}')
                    # Plot each subplot
                    axes[0].plot(ll_raw[i - 5:i + 5, :].ravel(), label='Data 1')
                    axes[0].axhline(y=selected_threshold, color='r', linestyle='-')
                    if (whichChannelHasSeizure[0] == 1):
                        axes[0].set_title("Seizure in this channel")

                    axes[1].plot(ll_predicted_1[i - 5:i + 5, :].ravel(), label='predicted')
                    axes[1].axhline(y=selected_threshold_predicted, color='r', linestyle='-')
                    if (whichChannelHasSeizure[0] == 1):
                        axes[1].set_title("Seizure in this channel")

                    plt.tight_layout()  # Adjust layout to prevent overlapping
                    plt.show()



def getThresholdsPerPatientAfterCleaning(path, patient_number, channel_number, sz_num):
    #sz_num = countNumberOfSeizuresPerPerson(patient_number)

    avg_list = []
    std_list = []
    max_list = []

    for sz in range(1, sz_num + 1):

        predicted_data_1 = np.load(os.path.join(path, f"pat_{patient_number}_sz_{sz}_ch_{channel_number}.npy"))

        ll = linelength(predicted_data_1)

        # ll = thetaBandPower(predicted_data_1)
        # ll = ll.squeeze(-1)

        avg = np.average(ll)
        std = np.std(ll)

        avg_list.append(avg)
        std_list.append(std)

        moving_average = movingAverage(ll)
        max_list.append(np.max(moving_average))

    avg_list = np.array(avg_list)
    std_list = np.array(std_list)

    avg = np.average(avg_list)
    std = np.sqrt(np.sum(std_list ** 2) / sz_num)

    max_ll = np.max(max_list)

    #thresholds = [avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std, avg + 4 * std, avg + 5 * std, avg + 6 * std]
    thresholds = [max_ll * 0.5, max_ll * 0.55, max_ll * 0.6, max_ll * 0.65, max_ll * 0.7, max_ll * 0.75, max_ll * 0.8, max_ll * 0.85, max_ll * 0.9, max_ll * 0.95, max_ll * 0.99]
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

def postProcessFP_ConsecValus_modified(true_label, predicted_label):

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

        for j in range(0, len(conf_mat_label) - 6):
            #and conf_mat_label[j + 3] == "FP" and conf_mat_label[j + 4] == "FP"
            if conf_mat_label[j] == "FP" and conf_mat_label[j + 1] == "FP" and conf_mat_label[j + 2] == "FP":
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

def postProcessFP_firingMethod(true_label, predicted_label):

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

        FP_count = 0
        k = 0
        while k < len(conf_mat_label) - 1:
            if predicted_label[k, i] == 1 and true_label[k] == 0:
                FP_count += 1
                k += 30
            else:
                k += 1
        FP_list.append(FP_count)

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

def getThresholdsPerPatientAfterCleaning_newThresholdMethod(path, patient_number, channel_number, sz_num):
    #sz_num = countNumberOfSeizuresPerPerson(patient_number)

    avg_list = []
    std_list = []

    ll_list = []
    label_list = []

    for sz in range(1, sz_num + 1):

        predicted_data_1 = np.load(os.path.join(path, f"pat_{patient_number}_sz_{sz}_ch_{channel_number}.npy"))

        path_extension_labels = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s"
        file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{sz}_labels.npy")
        labels = np.load(file_path_labels)

        ll = linelength(predicted_data_1)

        ll_list.append(ll)
        label_list.append(labels)

        # ll = thetaBandPower(predicted_data_1)
        # ll = ll.squeeze(-1)

    concatenated_ll = np.concatenate(ll_list)
    concatenated_labels = np.concatenate(label_list)


    fpr, tpr, thresholds = roc_curve(concatenated_labels, concatenated_ll)

    desired_tpr_values = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    # Initialize a dictionary to store thresholds corresponding to each desired TPR value
    thresholds_at_desired_tpr = []

    # Find thresholds for each desired TPR value
    for desired_tpr in desired_tpr_values:
        index = np.argmax(tpr >= desired_tpr)
        thresholds_at_desired_tpr.append(thresholds[index])

    return thresholds_at_desired_tpr


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

postProcessedFPList_new_new_post = []
postProcessedFPList_old_new_post = []
postProcessedFPList_45_new_post = []


for p in range(1, 51):

    #cProfile.run("countNumberOfSeizuresPerPerson(p)")
    sz_num = countNumberOfSeizuresPerPerson(p)

    conf_list = []
    conf_list_actual_data = []
    conf_list_45 = []

    # thresholds_old_ch_1 = getThresholdsPerPatient(p, 1, sz_num)
    # thresholds_old_ch_2 = getThresholdsPerPatient(p, 2, sz_num)
    # thresholds_old_ch_3 = getThresholdsPerPatient(p, 3, sz_num)
    # thresholds_old_ch_4 = getThresholdsPerPatient(p, 4, sz_num)

    thresholds_old_ch_1 = getThresholdsPerPatient(p, 1, sz_num)
    thresholds_old_ch_2 = getThresholdsPerPatient(p, 2, sz_num)
    thresholds_old_ch_3 = getThresholdsPerPatient(p, 3, sz_num)
    thresholds_old_ch_4 = getThresholdsPerPatient(p, 4, sz_num)


    #print("clean data thresholds: ", thresholds_old_ch_1, thresholds_old_ch_2, thresholds_old_ch_3, thresholds_old_ch_4)

    #path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_norm_method"
    path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering"
    thresholds_new_ch_1 = getThresholdsPerPatientAfterCleaning(path, p, 1, sz_num)
    thresholds_new_ch_2 = getThresholdsPerPatientAfterCleaning(path, p, 2, sz_num)
    thresholds_new_ch_3 = getThresholdsPerPatientAfterCleaning(path, p, 3, sz_num)
    thresholds_new_ch_4 = getThresholdsPerPatientAfterCleaning(path, p, 4, sz_num)
    #print("new data thresholds: ", thresholds_new_ch_1, thresholds_new_ch_2, thresholds_new_ch_3, thresholds_new_ch_4)


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
        # predicted_data_1 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize", f"pat_{p}_sz_{sz}_ch_1.npy"))
        # #predicted_data_1 = predicted_data_1.squeeze(-1)
        #
        # predicted_data_2 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize", f"pat_{p}_sz_{sz}_ch_2.npy"))
        # #predicted_data_2 = predicted_data_2.squeeze(-1)
        #
        # predicted_data_3 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize", f"pat_{p}_sz_{sz}_ch_3.npy"))
        # #predicted_data_3 = predicted_data_3.squeeze(-1)
        #
        # predicted_data_4 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize", f"pat_{p}_sz_{sz}_ch_4.npy"))
        # ##predicted_data_4 = predicted_data_4.squeeze(-1)
        #
        predicted_data_1 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering", f"pat_{p}_sz_{sz}_ch_1.npy"))
        #predicted_data_1 = predicted_data_1.squeeze(-1)

        predicted_data_2 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering", f"pat_{p}_sz_{sz}_ch_2.npy"))
        #predicted_data_2 = predicted_data_2.squeeze(-1)

        predicted_data_3 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering", f"pat_{p}_sz_{sz}_ch_3.npy"))
        #predicted_data_3 = predicted_data_3.squeeze(-1)

        predicted_data_4 = np.load(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering", f"pat_{p}_sz_{sz}_ch_4.npy"))
        ##predicted_data_4 = predicted_data_4.squeeze(-1)

        # predicted_data_1 = np.load(
        #     os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize",
        #                  f"pat_{p}_sz_{sz}_ch_1.npy"))
        # # predicted_data_1 = predicted_data_1.squeeze(-1)
        #
        # predicted_data_2 = np.load(
        #     os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize",
        #                  f"pat_{p}_sz_{sz}_ch_2.npy"))
        # # predicted_data_2 = predicted_data_2.squeeze(-1)
        #
        # predicted_data_3 = np.load(
        #     os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize",
        #                  f"pat_{p}_sz_{sz}_ch_3.npy"))
        # # predicted_data_3 = predicted_data_3.squeeze(-1)
        #
        # predicted_data_4 = np.load(
        #     os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby_normalize",
        #                  f"pat_{p}_sz_{sz}_ch_4.npy"))
        ##predicted_data_4 = predicted_data_4.squeeze(-1)

        # predicted_data_1 = np.load(os.path.join(
        #     r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering_lstm",
        #     f"pat_{p}_sz_{sz}_ch_1.npy"))
        # # predicted_data_1 = predicted_data_1.squeeze(-1)
        #
        # predicted_data_2 = np.load(os.path.join(
        #     r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering_lstm",
        #     f"pat_{p}_sz_{sz}_ch_2.npy"))
        # # predicted_data_2 = predicted_data_2.squeeze(-1)
        #
        # predicted_data_3 = np.load(os.path.join(
        #     r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering_lstm",
        #     f"pat_{p}_sz_{sz}_ch_3.npy"))
        # # predicted_data_3 = predicted_data_3.squeeze(-1)
        #
        # predicted_data_4 = np.load(os.path.join(
        #     r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering_lstm",
        #     f"pat_{p}_sz_{sz}_ch_4.npy"))
        # ##predicted_data_4 = predicted_data_4.squeeze(-1)


        ####when the predictions are not calculated yet
        # model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\lstm_encoder_bigger.h5")
        # model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
        # model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_checkPoint.h5")
        # model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\retrainWithEOG_LSTM.h5")
        # model = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_CNN_checkPoint.h5')
        #
        # # #
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
        # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering_cnn", f"pat_{p}_sz_{sz}_ch_1.npy"), predicted_data_1)
        # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering_cnn", f"pat_{p}_sz_{sz}_ch_2.npy"), predicted_data_2)
        # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering_cnn", f"pat_{p}_sz_{sz}_ch_3.npy"), predicted_data_3)
        # np.save(os.path.join(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering_cnn", f"pat_{p}_sz_{sz}_ch_4.npy"), predicted_data_4)







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
        #label_raw_data = doTwoAndOneOr(labels_1, labels_2, labels_3, labels_4)
        #label_raw_data = doAnds(labels_1, labels_2, labels_3, labels_4)

        innerFPWithPP_old = postProcessFP(label, label_raw_data)
        #innerFPWithPP_old = postProcessFP_firingMethod(label, label_raw_data)
        postProcessedFPList_old_new_post.append(postProcessFP_firingMethod(label, label_raw_data))
        postProcessedFPList_old.append(innerFPWithPP_old)



        labels_45_1 = getOnlyLabels(filteredSignal_1_45, label, thresholds_old_ch_1)
        labels_45_2 = getOnlyLabels(filteredSignal_2_45, label, thresholds_old_ch_2)
        labels_45_3 = getOnlyLabels(filteredSignal_3_45, label,  thresholds_old_ch_3)
        labels_45_4 = getOnlyLabels(filteredSignal_4_45, label, thresholds_old_ch_4)

        labels_45_data = doLogicalOR(labels_45_1, labels_45_2, labels_45_3, labels_45_4)
        #labels_45_data = doTwoAndOneOr(labels_45_1, labels_45_2, labels_45_3, labels_45_4)
        #labels_45_data = doAnds(labels_45_1, labels_45_2, labels_45_3, labels_45_4)

        # labels_45_data = np.logical_or(labels_45_1, labels_45_2)
        # labels_45_data = np.logical_or(labels_45_data, labels_45_3)
        # labels_45_data = np.logical_or(labels_45_data, labels_45_4)
        # labels_45_data = labels_45_data.astype(int)

        innerFPWithPP_45 = postProcessFP(label, labels_45_data)
        #innerFPWithPP_45 = postProcessFP_firingMethod(label, labels_45_data)
        postProcessedFPList_45_new_post.append(postProcessFP_firingMethod(label, labels_45_data))
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
        #predicted_label = doTwoAndOneOr(predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4)
        #predicted_label = doAnds(predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4)


        # predicted_label = np.logical_or(predicted_labels_1, predicted_labels_2)
        # predicted_label = np.logical_or(predicted_label, predicted_labels_3)
        # predicted_label = np.logical_or(predicted_label, predicted_labels_4)
        # predicted_label = predicted_label.astype(int)


        classified_at_least_once_new.append(classifiedAtLeastOnce(label, predicted_label))
        classified_at_least_once_10sec_new.append(classifiedAtLeastOnce_10sec(label, predicted_label))

        innerFPWithPP_new = postProcessFP(label, predicted_label)
        #innerFPWithPP_new = postProcessFP_firingMethod(label, predicted_label)
        postProcessedFPList_new_new_post.append(postProcessFP_firingMethod(label, predicted_label))
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

            #false_detections_predicted = getFlaseDetections(label, predicted_label, label_raw_data, selected_threshold)
            false_detections_predicted = getReducedFlaseDetections(label, predicted_label, label_raw_data, selected_threshold)
            #plotFalsePositives(false_detections_predicted, new_normalized_data_1, new_normalized_data_2, new_normalized_data_3, new_normalized_data_4, predicted_data_1, predicted_data_2, predicted_data_3, predicted_data_4, p, sz, selected_threshold, predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4)
            #plotFalsePositives_new(false_detections_predicted, new_normalized_data_1, new_normalized_data_2, new_normalized_data_3, new_normalized_data_4, predicted_data_1, predicted_data_2, predicted_data_3, predicted_data_4, p, sz, selected_threshold, predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4
            #                      ,labels_1, labels_2, labels_3, labels_4)
            improved_result = getImprovedResult(label, label_raw_data, predicted_label, selected_threshold)
            #print(f"channel thresholds new: channel 1: {thresholds_new_ch_1[selected_threshold]} channel 2: {thresholds_new_ch_2[selected_threshold]} channel 3: {thresholds_new_ch_3[selected_threshold]} channel 4: {thresholds_new_ch_4[selected_threshold]}")
            #print(f"channel thresholds old: channel 1: {thresholds_old_ch_1[selected_threshold]} channel 2: {thresholds_old_ch_2[selected_threshold]} channel 3: {thresholds_old_ch_3[selected_threshold]} channel 4: {thresholds_old_ch_4[selected_threshold]} ")

            #plotFalsePositives(improved_result, new_normalized_data_1, new_normalized_data_2, new_normalized_data_3, new_normalized_data_4, predicted_data_1, predicted_data_2, predicted_data_3, predicted_data_4, [thresholds_new_ch_1, thresholds_new_ch_2, thresholds_new_ch_3, thresholds_new_ch_4], [thresholds_old_ch_1, thresholds_old_ch_2, thresholds_old_ch_3, thresholds_old_ch_4], p, sz, selected_threshold, predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4)
            #plotLineLengthValues(improved_result, new_normalized_data_1, new_normalized_data_2, new_normalized_data_3, new_normalized_data_4, predicted_data_1, predicted_data_2, predicted_data_3, predicted_data_4, p, sz, selected_threshold, labels_1, labels_2, labels_3, labels_4)
            #dismproved_result = getdisImprovement(label, label_raw_data, predicted_label, selected_threshold)
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

ppofFP_new_array_new_post = np.array(postProcessedFPList_new_new_post)
ppofFP_new_array_new_post = np.mean(ppofFP_new_array_new_post, axis=0)

ppofFP_45_array_new_post = np.array(postProcessedFPList_45_new_post)
ppofFP_45_array_new_post = np.mean(ppofFP_45_array_new_post, axis=0)

ppofFP_old_array_new_post = np.array(postProcessedFPList_old_new_post)
ppofFP_old_array_new_post = np.mean(ppofFP_old_array_new_post, axis=0)



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


recall_list_new = averaged_list[:, 3] / (averaged_list[:, 3] + averaged_list[:, 2])
specificity_list_new = averaged_list[:, 0] / (averaged_list[:, 0] + averaged_list[:, 1])

recall_list_old = averaged_list_actual_data[:, 3] / (averaged_list_actual_data[:, 3] + averaged_list_actual_data[:, 2])
specificity_list_old = averaged_list_actual_data[:, 0] / (averaged_list_actual_data[:, 0] + averaged_list_actual_data[:, 1])

recall_list_45 = averaged_list_45[:, 3] / (averaged_list_45[:, 3] + averaged_list_45[:, 2])
specificity_list_45 = averaged_list_45[:, 0] / (averaged_list_45[:, 0] + averaged_list_45[:, 1])

# recall_list_new = classified_at_least_once_new / (classified_at_least_once_new + averaged_list[:, 2])
# specificity_list_new = averaged_list[:, 0] / (averaged_list[:, 0] + ppofFP_new_array_new_post)
#
# recall_list_old = classified_at_least_once_old / (classified_at_least_once_old + averaged_list_actual_data[:, 2])
# specificity_list_old = averaged_list_actual_data[:, 0] / (averaged_list_actual_data[:, 0] + ppofFP_old_array_new_post)
#
# recall_list_45 = classified_at_least_once_45 / (classified_at_least_once_45 + averaged_list_45[:, 2])
# specificity_list_45 = averaged_list_45[:, 0] / (averaged_list_45[:, 0] + ppofFP_45_array_new_post)



#
plt.plot(1 - specificity_list_new, recall_list_new)
plt.plot(1 - specificity_list_old, recall_list_old)
plt.plot(1 - specificity_list_45, recall_list_45)
plt.legend(["new", "old", "45"])
# auc_actual = auc(1-specificity_list_new, recall_list_new)
# auc_predicted = auc(1-specificity_list_old, recall_list_old)
# auc_45 = auc(1-specificity_list_45, recall_list_45)
#
# plt.xlabel("1-Specificity")
# plt.ylabel("Recall")
# legend_labels = ["new (AUC={:.2f})".format(auc_actual),
#                  "old (AUC={:.2f})".format(auc_predicted),
#                  "45 (AUC={:.2f})".format(auc_45)]
#
# plt.legend(legend_labels)
plt.show()

#
#
plt.plot(averaged_list[:, 1], classified_at_least_once_new)
plt.plot(averaged_list_actual_data[:, 1], classified_at_least_once_old)
plt.plot(averaged_list_45[:, 1], classified_at_least_once_45)
# auc_actual = auc(averaged_list_actual_data[:, 1], classified_at_least_once_new)
# auc_predicted = auc(averaged_list[:, 1], classified_at_least_once_old)
# auc_45 = auc(averaged_list_45[:, 1], classified_at_least_once_45)
plt.xlabel("False Positive Rate")
plt.ylabel("classified at least once as seizure")
# legend_labels = ["new (AUC={:.2f})".format(auc_actual),
#                  "old (AUC={:.2f})".format(auc_predicted),
#                  "45 (AUC={:.2f})".format(auc_45)]

#plt.legend(legend_labels)

plt.legend(["new", "old", "45"])
plt.show()
#




plt.plot(ppofFP_new_array, classified_at_least_once_new)
plt.plot(ppofFP_new_array_new_post, classified_at_least_once_new)

plt.plot(ppofFP_old_array, classified_at_least_once_old)
plt.plot(ppofFP_old_array_new_post, classified_at_least_once_old)

plt.plot(ppofFP_45_array, classified_at_least_once_45)
plt.plot(ppofFP_45_array_new_post, classified_at_least_once_45)

# auc_actual = auc(ppofFP_new_array, classified_at_least_once_new)
# auc_predicted = auc(ppofFP_old_array, classified_at_least_once_old)
# auc_45 = auc(ppofFP_45_array, classified_at_least_once_45)
plt.xlabel("False Positive Rate")
plt.ylabel("classified at least once as seizure- post processing")
# legend_labels = ["new (AUC={:.2f})".format(auc_actual),
#                  "old (AUC={:.2f})".format(auc_predicted),
#                  "45 (AUC={:.2f})".format(auc_45)]
#
# plt.legend(legend_labels)
plt.legend(["new", "new post", "old", "old post", "45", "45 post"])
plt.show()

plt.plot(ppofFP_new_array, classified_at_least_once_10sec_new)
plt.plot(ppofFP_old_array, classified_at_least_once_10sec_old)
plt.plot(ppofFP_45_array, classified_at_least_once_10sec_45)
auc_actual = auc(ppofFP_new_array, classified_at_least_once_10sec_new)
auc_predicted = auc(ppofFP_old_array, classified_at_least_once_10sec_old)
auc_45 = auc(ppofFP_45_array, classified_at_least_once_10sec_45)
plt.xlabel("False Positive Rate")
plt.ylabel("classified at least once as seizure in the first 10 seconds - post processing")
legend_labels = ["new (AUC={:.2f})".format(auc_actual),
                 "old (AUC={:.2f})".format(auc_predicted),
                 "45 (AUC={:.2f})".format(auc_45)]

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
#
#
# plt.plot(averaged_list[:, 1], averaged_list[:, 3])
# plt.plot(averaged_list_actual_data[:, 1], averaged_list_actual_data[:, 3])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
#
# plt.legend(["predicted", "actual data"])
# plt.show()



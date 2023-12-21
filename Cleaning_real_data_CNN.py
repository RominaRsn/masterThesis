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
import re

def linelength(data):
    data_diff = np.diff(data)
    return np.sum(np.absolute(data_diff), axis=1)

def extract_number(filename):
    # Use regular expression to find the number after "sz"
    match = re.search(r'sz_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN_bigger_kernel.h5")
model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"
#Get a list of files in the folder and sort them based on the number after "sz" in the name
files = sorted(os.listdir(folder_path), key=extract_number)
def countNumberOfSeizuresPerPerson(patient_number):
    # Iterate through all files in the folder
    cnt = 0
    for filename in os.listdir(folder_path):
        # Check if the file starts with "pat_1" and has the ".npy" extension
        if filename.startswith(f"pat_{patient_number}_") and filename.endswith(f"ch_1.npy"):
            # Construct the full path to the file

            cnt += 1
    return cnt

def normalize_ch_data(data1, data2, data3, data4):
    max_val = np.max([np.max(data1), np.max(data2), np.max(data3), np.max(data4)])
    min_val = np.min([np.min(data1), np.min(data2), np.min(data3), np.min(data4)])
    avg_val = (np.average(data1) + np.average(data2) + np.average(data3) + np.average(data4)) / 4

    data1 = (data1 - min_val) / (max_val - min_val)
    data1 = data1 - avg_val

    data2 = (data2 - min_val) / (max_val - min_val)
    data2 = data2 - avg_val

    data3 = (data3 - min_val) / (max_val - min_val)
    data3 = data3 - avg_val

    data4 = (data4 - min_val) / (max_val - min_val)
    data4 = data4 - avg_val

    return data1, data2, data3, data4


storage_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cheby"
for p in range(1, 51):
    sz_num = countNumberOfSeizuresPerPerson(p)
    for i in range(1, sz_num+1):
        file_path_1 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_1.npy")
        file_path_2 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_2.npy")
        file_path_3 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_3.npy")
        file_path_4 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_4.npy")

        data_1 = np.load(file_path_1)
        data_2 = np.load(file_path_2)
        data_3 = np.load(file_path_3)
        data_4 = np.load(file_path_4)

        data_1, data_2, data_3, data_4 = normalize_ch_data(data_1, data_2, data_3, data_4)


        #file_path = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_{ch_num}.npy")
        # data = np.load(file_path)
        # max_clean = np.max(data)
        # min_clean = np.min(data)
        # data_clean_normalized = (data - min_clean) / (max_clean - min_clean)
        # data_clean_normalized = data_clean_normalized - np.average(data_clean_normalized)
        result_1 = model.predict(data_1)
        result_2 = model.predict(data_2)
        result_3 = model.predict(data_3)
        result_4 = model.predict(data_4)

        s_path_1 = os.path.join(storage_path, f"pat_{p}_sz_{i}_ch_1.npy")
        s_path_2 = os.path.join(storage_path, f"pat_{p}_sz_{i}_ch_2.npy")
        s_path_3 = os.path.join(storage_path, f"pat_{p}_sz_{i}_ch_3.npy")
        s_path_4 = os.path.join(storage_path, f"pat_{p}_sz_{i}_ch_4.npy")

        np.save(s_path_1, result_1)
        np.save(s_path_2, result_2)
        np.save(s_path_3, result_3)
        np.save(s_path_4, result_4)

        # path = os.path.join(storage_path, f"cheby_gru_result_pat_{p}_sz_{i}_ch_{ch_num}.npy")
        # np.save(path, result)
        print(f"result_pat_{p}_sz_{i} results saved")





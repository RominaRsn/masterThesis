from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch
from keras.utils import plot_model
import os
import spicy
import tensorflow as tf
import pickle
import masterThesis.metrics as metrics
import neurokit2 as nk
import time
from sklearn import preprocessing
import os
import re
import pandas as pd
import masterThesis.feature_extraction as feature_extraction

def linelength(data):
    data_diff = np.diff(data)
    return np.sum(np.absolute(data_diff), axis=1)

def extract_number(filename):
    # Use regular expression to find the number after "sz"
    match = re.search(r'sz_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_more_filters.h5")

folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"
#Get a list of files in the folder and sort them based on the number after "sz" in the name
files = sorted(os.listdir(folder_path), key=extract_number)
def countNumberOfSeizuresPerPerson(patient_number, channel_number):
    # Iterate through all files in the folder
    cnt = 0
    for filename in os.listdir(folder_path):
        # Check if the file starts with "pat_1" and has the ".npy" extension
        if filename.startswith(f"pat_{patient_number}_") and filename.endswith(f"ch_{channel_number}.npy"):
            # Construct the full path to the file

            cnt += 1
    return cnt

#Iterate through the sorted files

storage_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_filtered"

for p in range(1, 2):
    for ch_num in range(1,5):
        sz_num = countNumberOfSeizuresPerPerson(p, ch_num)
        for i in range(1, sz_num+1):
            print(f"pat_{p}_sz_{i}_ch_{ch_num}")
            file_path = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_{ch_num}.npy")
            data = np.load(file_path)
            max_clean = np.max(data)
            min_clean = np.min(data)
            data_clean_normalized = (data - min_clean) / (max_clean - min_clean)
            data_clean_normalized = data_clean_normalized - np.average(data_clean_normalized)
            result = model.predict(data_clean_normalized)
            #print(f"result shape: {result.shape}")
            path = os.path.join(storage_path, f"result_pat_{p}_sz_{i}_ch_{ch_num}.npy")
            np.save(path, result)
            print(f"result_pat_{p}_sz_{i}_ch_{ch_num}.npy saved")

            old_ll = linelength(data_clean_normalized)
            new_ll = linelength(result.squeeze(-1))
            ll = np.vstack((old_ll, new_ll))
            #print(f"result shape: {ll.shape}")
            #
            path = os.path.join(storage_path, f"ll_pat_{p}_sz_{i}_ch_{ch_num}.npy")
            np.save(path, ll)








# for filename in files:
#     # Check if the file starts with "pat_1" and has the ".npy" extension
#     if filename.startswith("pat_1_") and filename.endswith("ch_1.npy"):
#         # Construct the full path to the file
#         file_path = os.path.join(folder_path, filename)
#
#         # Load the NumPy array from the file
#         data = np.load(file_path)
#
#         # Perform your operations with the loaded data
#         # For example, you can print the shape of the array
#         print(f"File: {filename}, Shape: {data.shape}")






#
#
# # Iterate through all files in the folder
# for filename in os.listdir(folder_path):
#     # Check if the file starts with "pat_1" and has the ".npy" extension
#     if filename.startswith("pat_1_") and filename.endswith("ch_1.npy"):
#         # Construct the full path to the file
#         file_path = os.path.join(folder_path, filename)
#
#         # Load the NumPy array from the file
#         data = np.load(file_path)
#
#         # Perform your operations with the loaded data
#         # For example, you can print the shape of the array
#         print(f"File: {filename}, Shape: {data.shape}")



# df = pd.read_csv(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s\pat_1_sz_1.csv")
#
# condition = df.iloc[:,0] == 1
# label = df[condition]
# label_np_1 = label["label"].to_numpy()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


#folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\new_filtered"
folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\real_data_filtering"
label_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s"

def label_to_binary(label, data):
    return_labels = []
    for i in range(0, label.shape[0]):
        if label[i] == 1:
            return_labels.append(np.ones(data.shape[1]))
        else:
            return_labels.append(np.zeros(data.shape[1]))

    return_labels = np.array(return_labels)
    return_labels = np.ravel(return_labels)
    return return_labels



def countNumberOfSeizuresPerPerson(patient_number):
    # Iterate through all files in the folder
    cnt = 0
    for filename in os.listdir(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"):
        # Check if the file starts with "pat_1" and has the ".npy" extension
        if filename.startswith(f"pat_{patient_number}_"):
            # Construct the full path to the file

            cnt += 1
    return cnt //4


for p in range(1,51):
    print(p)
    df_list = []
    a = countNumberOfSeizuresPerPerson(p)
    for sz_num in range(1, countNumberOfSeizuresPerPerson(p) + 1):
        ch1_data = np.load(f"{folder_path}/pat_{p}_sz_{sz_num}_ch_{1}.npy")
        ch2_data = np.load(f"{folder_path}/pat_{p}_sz_{sz_num}_ch_{2}.npy")
        ch3_data = np.load(f"{folder_path}/pat_{p}_sz_{sz_num}_ch_{3}.npy")
        ch4_data = np.load(f"{folder_path}/pat_{p}_sz_{sz_num}_ch_{4}.npy")

        ch1_data = ch1_data.reshape(ch1_data.shape[0], ch1_data.shape[1])
        ch2_data = ch2_data.reshape(ch2_data.shape[0], ch2_data.shape[1])
        ch3_data = ch3_data.reshape(ch3_data.shape[0], ch3_data.shape[1])
        ch4_data = ch4_data.reshape(ch4_data.shape[0], ch4_data.shape[1])

        seizure_id = sz_num
        seizure_block = pd.DataFrame({'ch1': np.ravel(ch1_data), 'ch2': np.ravel(ch2_data), 'ch3': np.ravel(ch3_data), 'ch4': np.ravel(ch4_data)})
        label_block = np.load(f"{label_path}/pat_{p}_sz_{sz_num}_labels.npy")
        labels = label_to_binary(label_block, ch1_data)

        seizure_block.set_index(pd.MultiIndex.from_arrays((seizure_id * np.ones_like(seizure_block.index), seizure_block.index)),inplace=True)
        print(seizure_block.shape)
        print(labels.shape)
        seizure_block['label'] = labels
        df_list.append(seizure_block)
    file_name = f"pat_{p}_results.csv"
    file_path_results = os.path.join(r"D:\RealData\export_eeg_emg", file_name)
    df = pd.concat(df_list, axis=0)
    df.to_csv(file_path_results)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import metrics as metrics
import keras
from keras.models import load_model
import neurokit2 as nk
import peakutils


folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"
path_extension_labels = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s"

model_ae = load_model(r"C:/Users/RominaRsn/PycharmProjects/MyMasterThesis/masterThesis/trained_models/ae_cheby_checkpoint.h5")
#model_ae = load_model( r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint.h5')
model_gru = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_checkpoint.h5")
model_cnn = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN_bigger_kernel.h5")
model_2 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\model_with_five_layers_more_filters.h5')

def countNumberOfSeizuresPerPerson (patient_number):
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


def are_consecutive(lst):
    return all(lst[i] + 1 == lst[i + 1] for i in range(len(lst) - 1))

model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5")
model_eog = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\retrainWithEOG_checkPoint.h5")


for p in range(40, 51):
    sz_num = countNumberOfSeizuresPerPerson(p)
    for i in range(1, sz_num+1):
            file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{i}_labels.npy")
            ##df = pd.read_csv(file_path_labels)
            ##condition = df.iloc[:, 0] == 1
            ##label = df[condition]
            #file_path = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_{ch_num}.npy")
            #data = np.load(file_path)

            label = np.load(file_path_labels)


            file_path_1 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_1.npy")
            file_path_2 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_2.npy")
            file_path_3 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_3.npy")
            file_path_4 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_4.npy")

            data_1 = np.load(file_path_1)
            data_2 = np.load(file_path_2)
            data_3 = np.load(file_path_3)
            data_4 = np.load(file_path_4)

            # data_1, data_2, data_3, data_4 = normalize_ch_data(data_1, data_2, data_3, data_4)
            #
            # data = np.empty_like(data_1)
            # if(ch_num == 1):
            #     data = data_1
            # elif(ch_num == 2):
            #     data = data_2
            # elif(ch_num == 3):
            #     data = data_3
            # elif(ch_num == 4):
            #     data = data_4

            mean_val = np.mean(data)
            std_val = np.std(data)

            # Normalize the data to the range [-1, 1]
            new_normalized_data = (data - mean_val) / std_val
            new_normalized_data = (new_normalized_data) / (np.max(new_normalized_data) - np.min(new_normalized_data))


            # num_zeros = (0, 12)
            # padded_data = np.pad(data, ((0, 0), num_zeros), mode='constant')
            # result_ae = model_ae.predict(new_normalized_data)
            # result_ae = result_ae.squeeze(-1)

            # result_2 = model_2.predict(new_normalized_data)
            # result_2 = result_2.squeeze(-1)

            # result_ae = result_ae.reshape(result_ae.shape[0], result_ae.shape[1])

            # result_gru = model_gru.predict(data)
            # result_gru = result_gru.reshape(result_gru.shape[0], result_gru.shape[1])
            #
            # result_cnn = model_cnn.predict(data)
            # result_cnn = result_cnn.reshape(result_cnn.shape[0], result_cnn.shape[1])

            #seeing the performance of the model on the EOG data
            result_eog = model_eog.predict(new_normalized_data)
            result_eog = result_eog.squeeze(-1)

            result_ae = model.predict(new_normalized_data)
            result_ae = result_ae.squeeze(-1)

            index_list = np.where(label == 1)[0]

            for i in index_list:

                if (i > 10):
                    # Create subplots with specified axes
                    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharey='col')
                    # Plot each subplot
                    axes[0].plot(new_normalized_data[i - 5:i + 5, :].ravel(), label='Data')
                    axes[0].set_title('Original data')
                    axes[0].set_ylabel('Signal amplitude')
                    axes[0].set_xlabel('Time')

                    axes[1].plot(result_eog[i - 5:i + 5, :].ravel(), label='EOG and EMG noise removed')
                    axes[1].set_title('EOG and EMG removed')
                    axes[1].set_ylabel('Signal amplitude')
                    axes[1].set_xlabel('Time')

                    axes[2].plot(result_ae[i - 5:i + 5, :].ravel(), label='EMG noise removed')
                    axes[2].set_title('EMG removed')
                    axes[2].set_ylabel('Signal amplitude')
                    axes[2].set_xlabel('Time')


                    plt.tight_layout()  # Adjust layout to prevent overlapping
                    plt.show()



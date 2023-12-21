import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import metrics as metrics
import keras
from keras.models import load_model
import neurokit2 as nk


folder_path = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data"
path_extension_labels = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s"

model_ae = load_model(r"C:/Users/RominaRsn/PycharmProjects/MyMasterThesis/masterThesis/trained_models/ae_cheby_checkpoint.h5")
model_gru = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\gru_checkpoint.h5")
model_cnn = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\deep_CNN_bigger_kernel.h5")

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


for p in range(14, 51):
    sz_num = countNumberOfSeizuresPerPerson(p)
    for i in range(1, sz_num+1):
        for ch_num in range(1,5):
            file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{i}_labels.npy")
            ##df = pd.read_csv(file_path_labels)
            ##condition = df.iloc[:, 0] == 1
            ##label = df[condition]
            label = np.load(file_path_labels)


            file_path_1 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_1.npy")
            file_path_2 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_2.npy")
            file_path_3 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_3.npy")
            file_path_4 = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_4.npy")

            data_1 = np.load(file_path_1)
            data_2 = np.load(file_path_2)
            data_3 = np.load(file_path_3)
            data_4 = np.load(file_path_4)

            data_1, data_2, data_3, data_4 = normalize_ch_data(data_1, data_2, data_3, data_4)

            data = np.empty_like(data_1)
            if(ch_num == 1):
                data = data_1
            elif(ch_num == 2):
                data = data_2
            elif(ch_num == 3):
                data = data_3
            elif(ch_num == 4):
                data = data_4


            result_ae = model_ae.predict(data)
            result_ae = result_ae.reshape(result_ae.shape[0], result_ae.shape[1])

            result_gru = model_gru.predict(data)
            result_gru = result_gru.reshape(result_gru.shape[0], result_gru.shape[1])

            result_cnn = model_cnn.predict(data)
            result_cnn = result_cnn.reshape(result_cnn.shape[0], result_cnn.shape[1])


            index = np.where(label ==1)
            for j in index[0]:
                input_data = np.array(data[j, :])
                input_data = input_data.reshape(input_data.shape[0], 1)
                #result_ae = model_ae.predict(input_data)
                #result_ae = result_ae.reshape(result_ae.shape[0], result_ae.shape[1])




                # # Plot the original data
                # plt.subplot(4, 1, 1)
                # plt.plot(input_data[0, :], label='Original Data')
                # plt.legend(loc='lower right')
                # plt.title('Original Data')
                #
                #
                # # Plot the result from model_ae
                # plt.plot(result_ae[0, :, 0], label='Model AE tResult')
                # plt.legend(loc='lower right')
                # plt.title('Model AE Result')
                #
                # # Plot the result from model_gru
                # plt.subplot(4, 1, 3, sharey=plt.gca())
                # plt.plot(result_gru[0, :, 0], label='Model GRU Result')
                # plt.legend(loc='lower right')
                # plt.title('Model GRU Result')
                #
                # # Plot the result from model_cnn
                # plt.subplot(4, 1, 4, sharey=plt.gca())
                #
                #
                # plt.plot(result_cnn[0, :, 0], label='Model CNN Result')
                # plt.legend(loc='lower right')
                # plt.title('Model CNN Result')
                #
                #
                # plt.suptitle(f'Patient {p}, Seizure {i}, Channel {ch_num}, Index {j}')
                #
                #
                #
                # plt.savefig(f'C:/Users/RominaRsn/Desktop/report_dec_15/figs/patient_{p}_seizure_{i}_channel_{ch_num}')

                # plt.tight_layout()  # Adjust layout for better spacing
                # #plt.show()
                fig, axes = plt.subplots(nrows=4, ncols=1, sharey='col')

                axes[0].plot(input_data, label='Original Data')
                axes[0].set_title('Original Data')
                axes[0].set_ylabel('Signal amplitude')
                axes[0].set_xlabel('Time')
                #plt.legend(loc='lower right')

                #print(smaller_reshaped_data_clean_test[row_index, :].shape)


                axes[1].plot(result_ae[j, :], label='Model AE Result')
                axes[1].set_title('Model AE Result')
                axes[1].set_ylabel('Signal amplitude')
                axes[1].set_xlabel('Time')
                #plt.legend(loc='lower right')

                #result = model.predict(result)
                #result = result.transpose()

                axes[2].plot(result_gru[j, :], label='Model GRU Result')
                axes[2].set_title('predicted data with gru')
                axes[2].set_ylabel('Signal amplitude')
                axes[2].set_xlabel('Time')
                #plt.legend(loc='lower right')

                axes[3].plot(result_cnn[j, :], label='Model CNN Result')
                axes[3].set_title('predicted data with cnn')
                axes[3].set_ylabel('Signal amplitude')
                axes[3].set_xlabel('Time')

                #plt.legend(loc='lower right')
                plt.suptitle(f'Patient {p}, Seizure {i}, Channel {ch_num}, Index {j}')
                plt.tight_layout()  # Adjust layout for better spacing

                plt.savefig(f'C:/Users/RominaRsn/Desktop/report_dec_15/figs/patient_{p}_seizure_{i}_channel_{ch_num}')


                #plt.show()
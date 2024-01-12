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

for p in range(14, 51):
    sz_num = countNumberOfSeizuresPerPerson(p)
    for i in range(1, sz_num+1):
        for ch_num in range(1,5):
            file_path_labels = os.path.join(path_extension_labels, f"pat_{p}_sz_{i}_labels.npy")
            ##df = pd.read_csv(file_path_labels)
            ##condition = df.iloc[:, 0] == 1
            ##label = df[condition]
            file_path = os.path.join(folder_path, f"pat_{p}_sz_{i}_ch_{ch_num}.npy")
            data = np.load(file_path)

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

            seizure = [];
            seizure_filtered = [];
            index = np.where(label ==1)
            index_list = index[0].tolist()

            a = are_consecutive(index_list)

            if(a == False):
                print(f"pat_{p}_sz_{i}_ch_{ch_num}")
                #seizure_filtered.append(result_2[j, :])
            # cnt = 0
            # for j in index_list:
            #     if(j != index_list[-1]):
            #         if(index_list.index(j) == index_list.index(j + 1) - 1):
            #             input_data = np.array(new_normalized_data[j, :])
            #             #base = peakutils.baseline(input_data, 50)
            #             input_data = input_data.reshape(input_data.shape[0], 1)
            #             seizure.append(input_data)
            #             #seizure_filtered.append(result_2[j, :])




            # seizure = np.array(seizure)
            # seizure_filtered = np.array(seizure_filtered)
            # seizure = seizure.ravel()
            # seizure_filtered = seizure_filtered.ravel()
            #
            # fig, axes = plt.subplots(nrows=2, ncols=1, sharey='col')
            #
            # axes[0].plot(seizure, label='Original Data')
            # axes[0].set_title('Original Data')
            # axes[0].set_ylabel('Signal amplitude')
            # axes[0].set_xlabel('Time')
            # # plt.legend(loc='lower right')
            #
            # # print(smaller_reshaped_data_clean_test[row_index, :].shape)
            #
            # # axes[1].plot(base, label='Model AE Result')
            # # axes[1].set_title('Model AE Result')
            # # axes[1].set_ylabel('Signal amplitude')
            # # axes[1].set_xlabel('Time')
            #
            # axes[1].plot(seizure_filtered, label='Model AE Result')
            # axes[1].set_title('Model AE Result')
            # axes[1].set_ylabel('Signal amplitude')
            # axes[1].set_xlabel('Time')
            # plt.legend(loc='lower right')
            # plt.show()


                #
                # # input_data = np.array(new_normalized_data[j, :])
                # # base = peakutils.baseline(input_data, 50)
                # # input_data = input_data.reshape(input_data.shape[0], 1)
                # #result_ae = model_ae.predict(input_data)
                # #result_ae = result_ae.reshape(result_ae.shape[0], result_ae.shape[1])
                #
                #
                #
                #
                # # # Plot the original data
                # # plt.subplot(4, 1, 1)
                # # plt.plot(input_data[0, :], label='Original Data')
                # # plt.legend(loc='lower right')
                # # plt.title('Original Data')
                # #
                # #
                # # # Plot the result from model_ae
                # # plt.plot(result_ae[0, :, 0], label='Model AE tResult')
                # # plt.legend(loc='lower right')
                # # plt.title('Model AE Result')
                # #
                # # # Plot the result from model_gru
                # # plt.subplot(4, 1, 3, sharey=plt.gca())
                # # plt.plot(result_gru[0, :, 0], label='Model GRU Result')
                # # plt.legend(loc='lower right')
                # # plt.title('Model GRU Result')
                # #
                # # # Plot the result from model_cnn
                # # plt.subplot(4, 1, 4, sharey=plt.gca())
                # #
                # #
                # # plt.plot(result_cnn[0, :, 0], label='Model CNN Result')
                # # plt.legend(loc='lower right')
                # # plt.title('Model CNN Result')
                # #
                # #
                # # plt.suptitle(f'Patient {p}, Seizure {i}, Channel {ch_num}, Index {j}')
                # #
                # #
                # #
                # # plt.savefig(f'C:/Users/RominaRsn/Desktop/report_dec_15/figs/patient_{p}_seizure_{i}_channel_{ch_num}')
                #
                # # plt.tight_layout()  # Adjust layout for better spacing
                # # #plt.show()
                # # fig, axes = plt.subplots(nrows=3, ncols=1, sharey='col')
                # #
                # # axes[0].plot(input_data, label='Original Data')
                # # axes[0].set_title('Original Data')
                # # axes[0].set_ylabel('Signal amplitude')
                # # axes[0].set_xlabel('Time')
                # # #plt.legend(loc='lower right')
                # #
                # # #print(smaller_reshaped_data_clean_test[row_index, :].shape)
                # #
                # #
                # # axes[1].plot(base, label='Model AE Result')
                # # axes[1].set_title('Model AE Result')
                # # axes[1].set_ylabel('Signal amplitude')
                # # axes[1].set_xlabel('Time')
                # #
                # # axes[2].plot(result_2[j, :], label='Model AE Result')
                # # axes[2].set_title('Model AE Result')
                # # axes[2].set_ylabel('Signal amplitude')
                # # axes[2].set_xlabel('Time')
                #
                # #plt.legend(loc='lower right')
                #
                # #result = model.predict(result)
                # #result = result.transpose()
                #
                # # axes[2].plot(result_gru[j, :], label='Model GRU Result')
                # # axes[2].set_title('predicted data with gru')
                # # axes[2].set_ylabel('Signal amplitude')
                # # axes[2].set_xlabel('Time')
                # # #plt.legend(loc='lower right')
                # #
                # # axes[3].plot(result_cnn[j, :], label='Model CNN Result')
                # # axes[3].set_title('predicted data with cnn')
                # # axes[3].set_ylabel('Signal amplitude')
                # # axes[3].set_xlabel('Time')
                #
                # #plt.legend(loc='lower right')
                # plt.suptitle(f'Patient {p}, Seizure {i}, Channel {ch_num}, Index {j}')
                # plt.tight_layout()  # Adjust layout for better spacing
                #
                # #plt.savefig(f'C:/Users/RominaRsn/Desktop/report_dec_15/figs/patient_{p}_seizure_{i}_channel_{ch_num}')
                #
                #
                # plt.show()
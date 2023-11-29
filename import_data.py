import pandas as pd
import numpy as np
import feature_extraction
import new_functions
import matplotlib.pyplot as plt
import neurokit2 as nk

data_path = r'D:\RealData\Data\export'
project = 'brainmep'
if project == 'brainmep':

    sampling_freq = 256

exam_window = 2    # in sec
stride = 1        # in sec
window_size = exam_window * sampling_freq
#stride_size = stride * sampling_freq
stride_size = 0
storagePath = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data'
storagePath_labels = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels_s'
old_interval = np.linspace(0, 512, 512, endpoint=False, dtype = int)
new_interval = np.linspace(0, 500, 500, endpoint=False, dtype = int)


def correctLength(x):
    result = []
    for i in range(0, x.shape[0]):
        result.append(nk.signal.signal_resample(x[i, :], desired_length=500, method='interpolation'))
    return np.array(result)

def correctLength_labels(x):
    result = []
    for i in range(0, x.shape[0]):
        result.append(nk.signal.signal_resample(x[i, :], desired_length=500, method='interpolation'))
    return np.array(result)
#
#
patients_list = range(1, 51)
# for p in patients_list:
#     patient_path = data_path + f'\\pat_{p}.csv'
#     data, labels = new_functions.import_csv(patient_path)
#     #
#     # for seizure_id, seizure_block in labels.groupby(level=0):
#     #     print(seizure_id, seizure_block.value_counts())
#
#
#     for seizure_id, seizure_block in data.groupby(level=0):
#         #print(seizure_block['labels'].value_counts())
#         seizure_block = np.array(seizure_block)
#         epoched_labels = new_functions.window_labels(labels, 512, stride_size)
#         numpy_array = epoched_labels.values
#         print(numpy_array.shape)
#         print(epoched_labels.shape)
#

        ##epoched_labels.to_csv(storagePath + f'\\labels_s\\pat_{p}_sz_{seizure_id}.csv')



        # for ch_num in range(0, data.shape[1]):
        #     epoched_data_tmp = feature_extraction.chunk_data(seizure_block[:, ch_num], window_size, stride_size)
        #     epoched_data_tmp_np = np.array(epoched_data_tmp)

            # resampled = correctLength(epoched_data_tmp_np)
            #
            # name_extension = f'pat_{p}_sz_{seizure_id}_ch_{ch_num + 1}'
            # save_path = storagePath + f'\\{name_extension}'
            # np.save(save_path, resampled)


    # start_index = 0
    # end_index = 0
    # for seizure_id, seizure_block in new_array[0].groupby(level=0):
    #
    #
    #     a = seizure_block.index
    #     b = seizure_block.index.get_level_values(1)
    #     print(a.size)
    #
    #     end_index = start_index + a.size
    #
    #     chunk_label = labels[start_index:end_index]
    #     chunk_label = np.array(chunk_label)
    #     chunk_label_tmp = feature_extraction.chunk_labels(chunk_label, window_size, stride_size)
    #     print(chunk_label_tmp.shape)
    #
    #     resampled_labels = correctLength(chunk_label_tmp)
    #
    #     if(1 in resampled_labels):
    #         plt.plot(chunk_label_tmp[0])
    #         plt.plot(resampled_labels[0])
    #         plt.show()
    #         print(seizure_id)
    #
    #     # storagePath = r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\labels"
    #     # name_extension = f'pat_{p}_sz_{seizure_id}_labels'
    #     # save_path = storagePath + f'\\{name_extension}'
    #     #
    #     # np.save(save_path, resampled_labels)
    #     start_index = end_index

#
#
def resample_block(x):
    result = []
    for i in range(0, x.shape[1]):
        result.append(nk.signal.signal_resample(x[:, i], sampling_rate=256, desired_sampling_rate=250 , method='interpolation'))
    result = np.array(result)
    return np.transpose(result)


def resample_block_labels(x):

    result = nk.signal.signal_resample(x, sampling_rate=256, desired_sampling_rate=250 , method='interpolation')
    result = np.array(result)
    return result




for p in patients_list:
    patient_path = data_path + f'\\pat_{p}.csv'
    data, labels = new_functions.import_csv(patient_path)
    for seizure_id, seizure_block in data.groupby(level=0):
        seizure_block = np.array(seizure_block)
        resampled_block = resample_block(seizure_block)

        for ch_num in range(0, data.shape[1]):
            epoched_data_tmp = feature_extraction.chunk_data(resampled_block[:, ch_num], 500, stride_size)

            print(f'ch_num{ch_num}, seizure_id:{seizure_id}, shape:{epoched_data_tmp.shape}')

            name_extension = f'pat_{p}_sz_{seizure_id}_ch_{ch_num + 1}'
            save_path = storagePath + f'\\{name_extension}'
            np.save(save_path, epoched_data_tmp)
            #plt.plot(epoched_data_tmp_np[0])
    print('-------------------------------------')
    for seizure_id, label_block in labels.groupby(level=0):
        label_block = np.array(label_block)
        resampled_labels = resample_block_labels(label_block)
        #labels_df = pd.DataFrame(resampled_labels, columns=['label'])
        epoched_labels = new_functions.window_labels(resampled_labels, 500, stride_size)
        print(f'seizure_id:{seizure_id}, shape:{epoched_labels.shape}')

        name_extension = f'pat_{p}_sz_{seizure_id}_labels'
        save_path = storagePath_labels + f'\\{name_extension}'
        np.save(save_path, epoched_labels)


        #np.save(f'pat_{p}_sz_{seizure_id}_ch_{ch_num}', epoched_data_tmp)


#

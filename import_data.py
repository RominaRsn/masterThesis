import pandas as pd
import numpy as np
import feature_extraction
import new_functions
import matplotlib.pyplot as plt
import neurokit2 as nk

project = 'brainmep'
if project == 'brainmep':

    data_path = r'C:\Users\RominaRsn\Desktop\Data\export'

    sampling_freq = 256

exam_window = 2    # in sec
stride = 1        # in sec
window_size = exam_window * sampling_freq
stride_size = stride * sampling_freq
storagePath = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data'

old_interval = np.linspace(0, 512, 512, endpoint=False, dtype = int)
new_interval = np.linspace(0, 500, 500, endpoint=False, dtype = int)


def correctLength(x):
    result = []
    for i in range(0, x.shape[0]):
        result.append(nk.signal.signal_resample(x[i, :], desired_length=500, method='interpolation'))
    return np.array(result)


patients_list = range(1,51)
for p in patients_list:
    patient_path = data_path + f'\\pat_{p}.csv'
    data, labels = new_functions.import_csv(patient_path) 
    new_array = new_functions.import_csv(patient_path)
    #
    # for seizure_id, seizure_block in labels.groupby(level=0):
    #     print(seizure_id, seizure_block.value_counts())


    for seizure_id, seizure_block in data.groupby(level=0):
        #print(seizure_block['labels'].value_counts())
        seizure_block = np.array(seizure_block)
        epoched_labels = new_functions.window_labels(labels, window_size, stride_size)
        epoched_labels.to_csv(storagePath + f'\\labels_s\\pat_{p}_sz_{seizure_id}.csv')



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
# for p in patients_list:
#     patient_path = data_path + f'\\pat_{p}.csv'
#     data, labels = new_functions.import_csv(patient_path)
#     for seizure_id, seizure_block in data.groupby(level=0):
#         seizure_block = np.array(seizure_block)
#         for ch_num in range(0, data.shape[1]):
#             epoched_data_tmp = feature_extraction.chunk_data(seizure_block[:, ch_num], window_size, stride_size)
#             epoched_labels_tmp = feature_extraction.chunk_labels(labels, window_size, stride_size)
#             epoched_data_tmp_np = np.array(epoched_data_tmp)
#
#             # epoched_labels_tmp_np = np.array(epoched_labels_tmp)
#             # resampled_labels = correctLength(epoched_labels_tmp_np)
#             chunked_label = labels[0:epoched_data_tmp_np.shape[0]]
#             ##chunked_label = np.array(chunked_label)
#             resampled = correctLength(epoched_data_tmp_np)
#             a = new_functions.load_labels()
#
#
#             print(resampled.shape)
#             name_extension = f'pat_{p}_sz_{seizure_id}_ch_{ch_num + 1}'
#             save_path = storagePath + f'\\{name_extension}'
#             np.save(save_path, resampled)
#             #plt.plot(epoched_data_tmp_np[0])
#
#
#             #np.save(f'pat_{p}_sz_{seizure_id}_ch_{ch_num}', epoched_data_tmp)
#
#
#

import numpy as np
import pandas as pd
import feature_extraction
import os, sys
import scipy.io



class read_patient_info_brainmep:
    def __init__(self, patient_path, design):
        self.patient_path = patient_path
        self.load_path = patient_path + '\\' + "training_data_2"
        self.save_path = patient_path + '\\brain_mep\\design_' + design + '\\'#########TODO----- FIX
        self.design = design

    def get_num_seizures(self):
        os.chdir(self.patient_path)
        seizure_count = int(scipy.io.loadmat('number_seizures.mat')['num_seiz'])
        #num_seizures = int(mat73.loadmat('number_seizures.mat')['num_seiz'])
        return seizure_count

    def get_channels_list(self):
        "get_channel_list() returns the list of channels."
        os.chdir(self.patient_path)
        ch_file = open('laplace_electrodes_1.txt', 'r+')
        chan_list = ch_file.read().split()
        return chan_list

    def load_data(self):
        patient_data = []
        for seizure_n in range(1, self.get_num_seizures() + 1, 1):
            try: 
                data = scipy.io.loadmat(self.load_path + f'\\seizure{seizure_n}.mat')['det_chs']
                
            except:
                print(f'    -Seizure {seizure_n} not found! Exiting...')
                sys.exit()
               
            data = pd.DataFrame(data, index=range(0,len(data)))
            data = data.set_index(pd.MultiIndex.from_arrays((seizure_n* np.ones_like(data.index), data.index)))
            patient_data.append(data)
            #patient_data['seizure' + f'{seizure_n}'] = data
            #print('Data size after adaptation: ' + f'{data.shape}')
        patient_data = pd.concat(patient_data)   
        patient_data.rename(columns = {0:'ch1', 1:'ch2',
                                       2:'ch3', 3:'ch4'}, inplace = True)
        print(f' - Patient data succesfully loaded - seizures: {self.get_num_seizures()}') 
        return patient_data

    def load_label(self):

        #if os.path.exists(self.save_path):
        #    print("There is already a directory for this design")
        #else:
        #    os.mkdir(self.save_path)
        #os.chdir(self.save_path)
        # if os.path.exists('labels'):
        #     pass
        # else:
        #     os.mkdir('labels')

        label_blocks = []

        for seizure_id in range(1, self.get_num_seizures() + 1, 1):
            try:
                seizure_labels = scipy.io.loadmat(self.load_path + f'\\seizure{seizure_id}_class_label.mat')['class_label']
                # data_labels = mat73.loadmat(self.load_path + '\\seizure' + str(sz_num) + '_class_label.mat')['class_label']
            except:
                print(f'    -Labels {seizure_id} not found! Exiting...')
                sys.exit()

            seizure_labels = np.int16(seizure_labels)
            seizure_labels = pd.DataFrame(seizure_labels, index=range(0,len(seizure_labels)))
            seizure_labels = seizure_labels.set_index(pd.MultiIndex.from_arrays((seizure_id* np.ones_like(seizure_labels.index), seizure_labels.index)))
            label_blocks.append(seizure_labels)

        sample_labels_df = pd.concat(label_blocks)  
        sample_labels_df.rename(columns={0:'label'}, inplace = True)
        sample_labels_df.replace(-1, 0, inplace=True)
            # # saving class labels
            # os.chdir(self.save_path + 'labels')
            # f = open('seizure' + str(sz_num) + '_class_label.txt', 'w')
            # # f.write(str(patient_label['seizure' + str(sz_num)][0].tolist()))
            # f.write(str(patient_label['seizure' + str(sz_num)].tolist()))
            # f.close()

        return sample_labels_df


def window_idx(raw_data, window_size, stride_size, update_data= False):
    # TODO check for 0 stride size
    #seizure_count = np.unique(raw_data.index.get_level_values(0))
    window_indices = [] 
    if stride_size != 0: 
        overlap_size = window_size - stride_size
    else: 
        overlap_size = 0
        stride_size = window_size
    
    #for n in seizure_count: # checking all seizures for overhang
    for seizure_id, data in raw_data.groupby(level=0):
        #data = raw_data.loc[n]
        #print(overlap_size, window_size)
        #print((data.shape[0] - window_size),(window_size - overlap_size) )
        num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
        overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)
        
        if overhang != 0 and update_data == True:  #TODO eliminating overhang
            overhang_mask = np.ones(len(data), dtype=bool)
            overhang_mask[-overhang:] = False
            data = data[overhang_mask]
            raw_data.loc[seizure_id] = data  # Update raw_data with overhang removed             
        
        for i in list(range(0, data.shape[0] - overlap_size, stride_size)): # iterating all window start indices
            indices = tuple([seizure_id, i, i + window_size - 1])  # defining end indices -> tuple(seizure number, start, end)
            window_indices.append(indices)

    if update_data:            
        return raw_data, window_indices 
    else: 
        return window_indices                # TODO UPDATE raw_data -------------------------------------------

def binarize_array(arr, keepdims = False):
    return np.any(arr, axis=1, keepdims=False).astype(int)


def window_labels(sample_label, window_size, stride_size):
    # block_list = []
    # for seizure_id, label_block in sample_label.groupby(level=0):
    #
    #     label_block= np.array(label_block)
    #     window_labels = feature_extraction.chunk_data(label_block, window_size, stride_size, flatten_inside_window=True)
    #     window_label = binarize_array(window_labels, keepdims = False)
    #     window_label = pd.DataFrame(window_label, columns= ['label'])
    #     window_label.set_index(pd.MultiIndex.from_arrays((seizure_id* np.ones_like(window_label.index), window_label.index)), inplace=True)
    #     block_list.append(window_label)
    #
    # labels_df= pd.concat(block_list, axis=0)
    #
    # return labels_df
    window_labels = feature_extraction.chunk_data(sample_label, window_size, stride_size, flatten_inside_window=True)
    window_label = binarize_array(window_labels, keepdims=False)
    #window_label = pd.DataFrame(window_label, columns=['label'])
    window_label = np.array(window_label)
    return window_label
    #window_label.set_index(pd.MultiIndex.from_arrays((seizure_id* np.ones_like(window_label.index), window_label.index)), inplace=True)


def import_csv(path):
    data = pd.read_csv(path, index_col=(0,1))
    labels =  data['label']
    data.drop(columns='label', inplace=True)
    return data, labels
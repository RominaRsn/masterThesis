import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


combo_model_result = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\ae_cnn_combo\result\result_pat_14_sz_2_ch_1.npy")
combo_model_result = combo_model_result.squeeze(-1)

data = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\real_data\pat_14_sz_2_ch_1.npy")
max_clean = np.max(data)
min_clean = np.min(data)
data_clean_normalized = (data - min_clean) / (max_clean - min_clean)
data_clean_normalized = data_clean_normalized - np.average(data_clean_normalized)

fig, axes = plt.subplots(nrows=2, ncols=1, sharey='col')
i = 1000
#row_index = np.random.randint(0, a)
#col_index = np.random.randint(0, 11520000/500)

axes[0].plot(data_clean_normalized[i, :], label = 'Real Data')
axes[0].set_title('Real data')
axes[0].set_ylabel('Signal amplitude')
axes[0].set_xlabel('Time')

#print(smaller_reshaped_data_clean_test[row_index, :].shape)


axes[1].plot(combo_model_result[i, :], label = 'cleaned Data')
axes[1].set_title('cleaned data')
axes[1].set_ylabel('Signal amplitude')
axes[1].set_xlabel('Time')

plt.legend()
plt.show()

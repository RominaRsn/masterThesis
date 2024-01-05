import numpy as np
import matplotlib.pyplot as plt


data_clean = np.load(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_0.npy')
data_noisy = np.load(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_0.npy')

max_clean = np.max(data_clean)
max_noisy = np.max(data_noisy)
min_clean = np.min(data_clean)
min_noisy = np.min(data_noisy)
max_val = max(max_clean, max_noisy)
min_val = min(min_clean, min_noisy)


######normalize between [-avg, 1-avg]
# data_clean_normalized = (data_clean - min_val) / (max_val - min_val)
# data_noisy_normalized = (data_noisy - min_val) / (max_val - min_val)
# avg = (np.mean(data_clean_normalized) + np.mean(data_noisy_normalized))/2
#
# data_clean_normalized = data_clean_normalized - avg
# data_noisy_normalized = data_noisy_normalized - avg
#
# np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy', data_clean_normalized)
# np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy', data_noisy_normalized)
#
# val = max_val - min_val
# print(val)
# plt.plot(data_clean_normalized[0, :])
# plt.plot(data_noisy_normalized[0, :])
# plt.plot(data_clean[0, :])
# plt.plot(data_noisy[0, :])
# plt.legend(['clean_normalized', 'noisy_normalized', 'clean', 'noisy'])
#
# plt.show()

# # normalizing to [-1, 1] and saving to 32*32 matrix
# data_clean_normalized = ((data_clean - min_val) / (max_val - min_val)) * 2 - 1
# data_noisy_normalized = ((data_noisy - min_val) / (max_val - min_val)) * 2 - 1
#
# # pad with zeros
# num_zeros_to_add = 524
# data_clean_normalized = np.pad(data_clean_normalized, ((0, 0), (0, num_zeros_to_add)), 'constant')
# data_noisy_normalized = np.pad(data_noisy_normalized, ((0, 0), (0, num_zeros_to_add)), 'constant')
#
# # reshape to 32*32 matrix
# reshaped_data_clean_normalized = data_clean_normalized.reshape((data_clean_normalized.shape[0], 32, 32))
# reshaped_data_noisy_normalized = data_noisy_normalized.reshape((data_noisy_normalized.shape[0], 32, 32))
#
# #avg = (np.mean(data_clean_normalized) + np.mean(data_noisy_normalized))/2
# np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_resNet.npy', reshaped_data_clean_normalized)
# np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_resNet.npy', reshaped_data_noisy_normalized)

data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_resNet.npy")
data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_resNet.npy")

print("data loaded")

expanded_clean = np.repeat(data_clean_normalized_cheby[..., np.newaxis], 3, axis=-1)
expanded_noisy = np.repeat(data_noisy_normalized_cheby[..., np.newaxis], 3, axis=-1)

np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_resNet_expanded.npy', expanded_clean)
np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_resNet_expanded.npy', expanded_noisy)

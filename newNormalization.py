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

data_clean_normalized = (data_clean - min_val) / (max_val - min_val)
data_noisy_normalized = (data_noisy - min_val) / (max_val - min_val)

np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy', data_clean_normalized)
np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy', data_noisy_normalized)

val = max_val - min_val
print(val)
plt.plot(data_clean_normalized[0, :])
plt.plot(data_noisy_normalized[0, :])
plt.plot(data_clean[0, :])
plt.plot(data_noisy[0, :])
plt.legend(['clean_normalized', 'noisy_normalized', 'clean', 'noisy'])

plt.show()

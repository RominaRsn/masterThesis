import numpy as np
import matplotlib.pyplot as plt



data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")

plt.plot(data_clean_normalized[0, :])
plt.plot(data_noisy_normalized[0, :])
plt.legend(['clean_normalized', 'noisy_normalized', 'clean', 'noisy'])

plt.show()

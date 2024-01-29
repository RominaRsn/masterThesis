import matplotlib.pyplot as plt
import numpy as np
import mne
import neurokit2 as nk


clean_data = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\clean_data_eog.npy")


eog = np.load(r"C:\Users\RominaRsn\Downloads\EEGdenoiseNet-master\data\EOG_all_epochs.npy")

filtered_eog = []
for i in range(0, eog.shape[0]):
    filtered_eog.append(nk.signal_filter(eog[i, :], lowcut=25, highcut=120, sampling_rate=256))

filtered_eog = np.array(filtered_eog)

clean_std = np.std(clean_data)
std_eog = np.std(filtered_eog, axis=1)

scaled = []
for i in range(0, filtered_eog.shape[0]):
    scaled.append(filtered_eog[i, :] * (clean_std / std_eog[i]))

scaled = np.array(scaled)

resampled = []
for i in range(0, filtered_eog.shape[0]):
    resampled.append(nk.signal_resample(scaled[i, :], sampling_rate=256, desired_sampling_rate=250))

resampled = np.array(resampled)

noisy_data = np.zeros(clean_data.shape)
for i in range(0, clean_data.shape[0]):
    multiplicant = np.random.randint(2,5)
    index = np.random.randint(0, resampled.shape[0])
    noisy_data[i, :] = clean_data[i, :] + (resampled[index, :] * multiplicant)

np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\EOG_data\noisy_data_eog.npy", noisy_data)
#
# plt.plot(eog[0])
# plt.show()
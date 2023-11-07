import matplotlib.pyplot as plt
import numpy as np
import mne
import time
from sklearn import preprocessing

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

print(raw.info)

EEG_Clean1 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch3.vhdr");
signals = mne.ioraw_data = EEG_Clean1.get_data()
sampling_freq = 250

print(signals[1])
plt.plot(signals[1])
#plt.show()

def loadclean():
    EEG_Clean1 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch1.vhdr")
    EEG_Clean2 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch2.vhdr")
    EEG_Clean3 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch3.vhdr")
    signals = mne.ioraw_data = EEG_Clean1.get_data()
    clean_data2 = mne.ioraw_data = EEG_Clean2.get_data()
    signals = np.append(signals, clean_data2, axis=1)
    clean_data2 = mne.ioraw_data = EEG_Clean3.get_data()
    signals = np.append(signals, clean_data2, axis=1)
    sampling_freq = 250
    return (signals, sampling_freq)


def loadnoisy():
    EEG_NoisyF1 = mne.io.read_raw_brainvision(
        r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_1.vhdr")
    EEG_NoisyF2 = mne.io.read_raw_brainvision(
        r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_2.vhdr")
    EEG_NoisyF3 = mne.io.read_raw_brainvision(
        r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_3.vhdr")

    noisy_signals = mne.ioraw_data = EEG_NoisyF1.get_data()
    noisy_dataF4 = mne.ioraw_data = EEG_NoisyF2.get_data()
    noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)

    noisy_dataF4 = mne.ioraw_data = EEG_NoisyF3.get_data()
    noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)
    sampling_freq = 250
    return (noisy_signals, sampling_freq)


# Model configuration
input_shape = (500, 1)
batch_size = 32
#no_epochs = 30
no_epochs = 1


train_test_split = 0.4
validation_split = 0.1
verbosity = 1
max_norm_value = 6.0


start_time = time.time()
#load EEG
clean_EEG = loadclean()
#clean_EEG = loadsmallclean()
clean_data = clean_EEG[0]
sampling_freq = clean_EEG[1]

#noisy_data = loadnoisy()
noisy_data = loadnoisy()
#noisy_data = loadsmallnoisy()

noisy_dataF3 = noisy_data[0]


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time 1: {elapsed_time} seconds")
num_chanClean, num_SamplesClean = clean_data.shape
nChan = num_chanClean
num_chanNoisy, num_SamplesNoisy = noisy_dataF3.shape

val_clean = []
val_noisy = []

start_time = time.time()

for i in range(0, 180):
    for j in range(0, 23040):
        val_clean.append(clean_data[i, j * 2 * sampling_freq : (j + 1) * 2 * sampling_freq])
        val_noisy.append(noisy_dataF3[i, j * 2 * sampling_freq : (j + 1) * 2 * sampling_freq])


data_cleaned = np.array(val_clean)
data_noisy = np.array(val_noisy)


small_data_cleaned = data_cleaned[0:1000, :]
small_data_noisy = data_noisy[0:1000, :]
np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\small_data_cleaned.npy', small_data_cleaned)
np.save(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\small_data_noisy.npy', small_data_noisy)


data_cleaned_normalized = preprocessing.normalize(data_cleaned)
data_noisy_normalized = preprocessing.normalize(data_noisy)


#normalize data

# Specify the file path
file_path_clean = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_0.npy'
file_path_noisy = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_0.npy'
# Save the matrix to a NumPy binary file
np.save(file_path_clean, data_cleaned)
np.save(file_path_noisy, data_noisy)



start_time = time.time()
# Specify the file path
file_path_clean = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized.npy'
file_path_noisy = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized.npy'
# Save the matrix to a NumPy binary file
np.save(file_path_clean, data_cleaned_normalized)
np.save(file_path_noisy, data_noisy_normalized)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Normalization time: {elapsed_time} seconds")

print(data_cleaned_normalized.shape)
print(data_noisy_normalized.shape)


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

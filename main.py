import matplotlib.pyplot as plt
import numpy as np
import mne

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
    EEG_NoisyF1 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch1.vhdr")
    EEG_NoisyF2 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch2.vhdr")
    EEG_NoisyF3 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch3.vhdr")

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

#load EEG
clean_EEG = loadclean()
#clean_EEG = loadsmallclean()
clean_data = clean_EEG[0]
sampling_freq = clean_EEG[1]

#noisy_data = loadnoisy()
noisy_data = loadnoisy()
#noisy_data = loadsmallnoisy()

noisy_dataF3 = noisy_data[0]
num_chanClean, num_SamplesClean = clean_data.shape
nChan = num_chanClean
num_chanNoisy, num_SamplesNoisy = noisy_dataF3.shape
print("********************")
print(clean_data.shape)
print(num_chanClean)
print(num_SamplesClean)


# check dimensions of clean and noisy datafiles
if num_chanClean != num_chanNoisy:
    exit()

if num_SamplesClean != num_SamplesNoisy:
    exit()

pmaxn = np.max(noisy_dataF3)
pminn = np.min(noisy_dataF3)

#pmaxn = np.max(clean_data)
#pminn = np.min(clean_data)

#estimate value range for noisy and clean data
pure_probe = np.zeros((1, sampling_freq))
noisy_probe = np.zeros((1, sampling_freq))
nOffset = sampling_freq/2

for s in range(0, sampling_freq-1):
    pure_probe[0, s] = clean_data[0, s + 120]

for s in range(0, sampling_freq-1):
    noisy_probe[0, s] = noisy_dataF3[0, s + 120]

minc = 3 * np.min(pure_probe)
maxc = 3 * np.max(pure_probe)

minn = 3 * np.min(noisy_probe)
maxn = 3 * np.max(noisy_probe)

#determine number of epochs with data
ns = 128 * 3
#ns = 128
# for small dataset ns = 128
#ns=64
nb = 30

nDataEpochs = ns * nb * num_chanClean
#y_val_noisy = data_noisy[:,1]
y_val_noisy = np.zeros((nDataEpochs,  2 * sampling_freq))
noisy_sample = np.zeros((2 * sampling_freq, 1))
#y_val_pure = data_pure[:,1]
y_val_pure = np.zeros((nDataEpochs, 2 * sampling_freq))
pure_sample = np.zeros((2 * sampling_freq, 1))
# Reshape data
y_val_noisy_r = []
y_val_pure_r = []
# for i in range(0, len(y_val_noisy)):
index = 0;
nSeizureStartOffset = 60 * sampling_freq
stddnoisy = 1.e-25
stddpure = 1.e-25
pmaxn = 1.e-25
pmax = 1.e-25
pminn= -100000000000
pmin = -100000000000

#Number of times to go over each row to complete the row of 11520000 columns
n_s = np.int_(11520000/500)


print(sampling_freq)

val_data = np.zeros((23040, 0))

for i in range(0, n_s):
    for row in range(0, 180):

        #rand_index = np.random.randint(0, 11520000 - 500)
        val_data[i] = clean_data[row , n_s * 2 * sampling_freq : n_s * 2 * sampling_freq + 500]

val_data = np.array(val_data)
print(val_data.shape)
#print(signals)
#plt.plot(signals[1:5])
#plt.show()
#raw.compute_psd(fmax=50).plot(picks="data", exclude="bads")
#raw.plot(duration=5, n_channels=30)
#plt.show()
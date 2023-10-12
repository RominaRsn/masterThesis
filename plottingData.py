import matplotlib.pyplot as plt
import mne
import numpy as np


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
    EEG_NoisyF1 = mne.io.read_raw_brainvision(r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_1.vhdr")
    EEG_NoisyF2 = mne.io.read_raw_brainvision(r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_2.vhdr")
    EEG_NoisyF3 = mne.io.read_raw_brainvision(r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_3.vhdr")

    noisy_signals = mne.ioraw_data = EEG_NoisyF1.get_data()
    noisy_dataF4 = mne.ioraw_data = EEG_NoisyF2.get_data()
    noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)

    noisy_dataF4 = mne.ioraw_data = EEG_NoisyF3.get_data()
    noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)
    sampling_freq = 250
    return (noisy_signals, sampling_freq)


#load EEG
clean_EEG = loadclean()
#clean_EEG = loadsmallclean()
clean_data = clean_EEG[0]
sampling_freq = clean_EEG[1]

#noisy_data = loadnoisy()
noisy_data = loadnoisy()
#noisy_data = loadsmallnoisy()

noisy_dataF3 = noisy_data[0]


y_axis = np.linspace(0, 2 * sampling_freq)
for i in range(0,5):
    fig, axes = plt.subplots(nrows=2, ncols=1)


    row_index = np.random.randint(0, len(clean_data))
    col_index = np.random.randint(0, 11520000/500)

    axes[0].plot(clean_data[row_index, col_index : col_index + 500], label = 'Clean Data')
    axes[0].set_title('Clean data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')


    axes[1].plot(noisy_dataF3[row_index, col_index : col_index + 500], label = 'Noisy Data')
    axes[1].set_title('Noisy data')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')

    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of clean and noisy data')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

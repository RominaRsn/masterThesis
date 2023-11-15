import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")




# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)




# Assuming 'noisy_test' is your input signal
# Assuming sampling rate is 250 Hz
sampling_rate = 250
lowcut = 0.1
highcut = 45
order = 4

# Normalize cutoff frequencies to Nyquist frequency
lowcut_normalized = lowcut / (0.5 * sampling_rate)
highcut_normalized = highcut / (0.5 * sampling_rate)

# Design Butterworth filter
b, a = butter(order, [lowcut_normalized, highcut_normalized], btype='band')

# Apply the filter using filtfilt
filteredSignal_45 = filtfilt(b, a, noisy_test)

b, a = butter(order, [lowcut_normalized, 30 / (0.5 * sampling_rate)], btype='band')
filteredSignal_30 = filtfilt(b, a, noisy_test)

b, a = butter(order, [lowcut_normalized, 70 / (0.5 * sampling_rate)], btype='band')
filteredSignal_70 = filtfilt(b, a, noisy_test)





signalIndexVector = [0, 1, 3, 7, 11, 13, 14, 16, 17]
for i in signalIndexVector:
    fig, axes = plt.subplots(nrows=5, ncols=1, sharey='col')

    row_index = i
    #row_index = np.random.randint(0, a)
    #col_index = np.random.randint(0, 11520000/500)

    axes[0].plot(clean_test[row_index, :], label = 'Clean Data')
    axes[0].set_title('Clean data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')

    #print(smaller_reshaped_data_clean_test[row_index, :].shape)


    axes[1].plot(noisy_test[row_index, :], label = 'Noisy Data')
    axes[1].set_title('Noisy data')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')

    #result = model.predict(result)
    #result = result.transpose()

    axes[2].plot(filteredSignal_30[row_index, :], label='predicted data with 3 layers in encoder')
    axes[2].set_title('highcut 30 Hz')
    axes[2].set_ylabel('Signal amplitude')
    axes[2].set_xlabel('Time')


    axes[3].plot(filteredSignal_45[row_index, :], label ='predicted data with 4 layers in encoder')
    axes[3].set_title('highcut 45 Hz')
    axes[3].set_ylabel('Signal amplitude')
    axes[3].set_xlabel('Time')

    axes[4].plot(filteredSignal_70[row_index, :], label ='filtered signal-45')
    axes[4].set_title('highcut 70 Hz')
    axes[4].set_ylabel('Signal amplitude')
    axes[4].set_xlabel('Time')

    # Add overall title
    fig.suptitle('Comparison of different butter worth filters', fontsize=16)

    plt.show()

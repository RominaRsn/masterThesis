import numpy as np
import matplotlib.pyplot as plt
import spicy
from sklearn.model_selection import train_test_split


data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
#data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")

signalIndexVector = [0, 1, 3, 7, 11, 13, 14, 16, 17]
for i in signalIndexVector:
    # array = spicy.signal.find_peaks(data_clean_normalized[i, :], distance=50)
    # print(array)

    # if(len(array) > 0):
    #     print("found peaks: ", i)
    plt.xticks(np.arange(0, 501, step=50))
    plt.plot(data_clean_normalized[i, :])
    plt.title(i)
    plt.show()

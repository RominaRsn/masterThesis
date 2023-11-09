import numpy as np
import matplotlib.pyplot as plt
import spicy


data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
#data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")


for i in range(10, 20):
    # array = spicy.signal.find_peaks(data_clean_normalized[i, :], distance=50)
    # print(array)

    # if(len(array) > 0):
    #     print("found peaks: ", i)
    plt.xticks(np.arange(0, 501, step=50))
    plt.plot(data_clean_normalized[i, :])
    plt.title(i)
    plt.show()

from scipy.signal import butter,filtfilt,iirnotch
import numpy as np

class metrics:
    # Class attribute
    # Constructor method (called when an object is created)

    def filtering_signals(data, fs, low_freq, high_freq, notch_freq, order):
        low_wn = low_freq / (fs * 0.5)
        high_wn = high_freq / (fs * 0.5)
        b, a = butter(order, low_wn)
        data = filtfilt(b, a, data, axis=0)
        #  b, a = butter(order, high_wn,'high')
        #  data = filtfilt(b, a, data, axis=0)
        #  b, a = iirnotch(notch_freq, 35, fs)
        # data = filtfilt(b, a, data, axis=1)
        # data = filtfilt(b, a, data, axis=0)
        return data

    def snr(firstSignal, secondSignal):
        snr = 10 * np.log10(np.sum(secondSignal ** 2) / np.sum((firstSignal - secondSignal) ** 2))
        return snr


    def snrDiff(self, originalSignal, preprocessedSignal, predictedSignal):
        snrInput =  self.snr(originalSignal, preprocessedSignal)
        snrOutput = self.snr(predictedSignal, preprocessedSignal)
        snrDiff = snrOutput - snrInput

        return snrDiff

    # Pearson correlation coefficient

    def accMetric(self, trueDataChannel, predictedDataChannel):
        dataChannel = np.vstack((trueDataChannel.T, predictedDataChannel.T))
        covarianceMatrix = np.cov(dataChannel)
        covariance = covarianceMatrix[0, 1]
        varianceTrue = covarianceMatrix[0, 0]
        variancePred = covarianceMatrix[1, 1]
        acc = covariance / np.sqrt(varianceTrue * variancePred)

        return acc

    # Relative RMSE

    def rrmseMetric(yTrue, yPred):
        yDiff = yPred - yTrue
        rmsYDiff = np.sqrt(np.mean(yDiff ** 2))
        rmsYTrue = np.sqrt(np.mean(yTrue ** 2))

        return rmsYDiff / rmsYTrue


# Creating an object of the class
example_object = metrics()

# Using the class methods

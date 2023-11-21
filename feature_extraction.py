import scipy.signal
import scipy.stats
from sklearn import preprocessing
import sklearn   
from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
import antropy
from statsmodels import robust
import pandas as pd


def chunk_labels(labels, window_size, stride_size, flatten_inside_window=True):
    # data= np.array(data)
    # assert labels.ndim == 1 or labels.ndim == 2
    # if labels.ndim == 1:
    #     labels = labels.reshape((-1, 1))

    if stride_size != 0:
        overlap_size = window_size - stride_size
    else:
        overlap_size = 0
        stride_size = window_size

        # get the number of overlapping windows that fit into the data
    num_windows = (labels.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = labels.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    # if overhang != 0:
    #     num_windows += 1
    #     newdata = np.zeros((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1]))
    #     newdata[:data.shape[0]] = data
    #     data = newdata

    sz = labels.dtype.itemsize
    ret = ast(
        labels,
        shape=(num_windows, window_size),
        strides=((window_size - overlap_size) * 1 * sz, sz)
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))
    return ret



def chunk_data(data, window_size, stride_size, flatten_inside_window=True):
    #data= np.array(data)
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))
    
    if stride_size != 0: 
        overlap_size = window_size - stride_size
    else: 
        overlap_size = 0
        stride_size = window_size    
   
    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    # if overhang != 0:
    #     num_windows += 1
    #     newdata = np.zeros((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1]))
    #     newdata[:data.shape[0]] = data
    #     data = newdata

    sz = data.dtype.itemsize
    ret = ast(
        data,
        shape=(num_windows, window_size * data.shape[1]),
        strides=((window_size - overlap_size) * data.shape[1] * sz, sz)
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))

    
def linelength(data):
    data_diff = np.diff(data)
    return np.sum(np.absolute(data_diff), axis=1)


def weighted_mean_freq(data, win_size, sampling_freq):
    analytic_signal = scipy.signal.hilbert(data, axis=1)  # [:,None]
    # print 'Hilbert transform size: ' + str(analytic_signal.shape)
    instantaneous_amp = np.abs(analytic_signal)
    # print 'instantaneous amplitude size: ' + str(instantaneous_amp.shape)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_freq)
    # print 'instantaneous frequency size: ' + str(instantaneous_frequency.shape)
    nominator_acc = np.zeros(2, )
    denominator_acc = np.zeros(2, )
    for inx in range(0, win_size - 1):
        nominator = np.multiply(instantaneous_amp[:, inx + 1], (instantaneous_frequency[:, inx] ** 2))
        denominator = np.multiply(instantaneous_amp[:, inx + 1], instantaneous_frequency[:, inx])
        nominator_acc = nominator_acc + nominator
        denominator_acc = denominator_acc + denominator
    return np.divide(nominator_acc, denominator_acc)


# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)


def time_corr(data, win_size):
    num_win = data.shape[1] / win_size
    # number of corr_coefficients = ((num_chan-1)*num_chan)/2 , number of eigenvalues= num_chan
    num_feat = ((data.shape[0] * (data.shape[0] - 1)) / 2) + data.shape[
        0]  # num_feat= numb of corr_coeff + numb of eigenvalues
    corr_feat = np.zeros((0, num_feat))
    for win_num in range(0, num_win):
        data_win = data[:, (win_num * win_size):(win_num + 1) * win_size]
        corr_matrix = np.corrcoef(data_win)
        eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
        corr_coefficients = upper_right_triangle(corr_matrix)  # custom func
        corr_val = np.expand_dims(np.append(corr_coefficients, eigenvalues), axis=0)
        corr_feat = np.concatenate((corr_feat, corr_val), axis=0)  # ,np.reshape(data_win,(1,win_size)),axis=0)
    return corr_feat


def fft(time_data):
    return np.log10(np.absolute(np.fft.rfft(time_data, axis=1)[:, 1:48]))


def freq_corr(data, win_size):
    num_win = data.shape[1] / win_size
    # number of corr_coefficients = ((num_chan-1)*num_chan)/2 , number of eigenvalues= num_chan
    num_feat = ((data.shape[0] * (data.shape[0] - 1)) / 2) + data.shape[
        0]  # num_feat= numb of corr_coeff + numb of eigenvalues
    fcorr_feat = np.zeros((0, num_feat))
    for win_num in range(0, num_win):
        data_win = data[:, (win_num * win_size):(win_num + 1) * win_size]
        fft_data = fft(data_win)
        scaled = sklearn.preprocessing.scale(fft_data, axis=0)
        corr_matrix = np.corrcoef(scaled)
        eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
        eigenvalues.sort()
        corr_coefficients = upper_right_triangle(corr_matrix)  # custom func
        fcorr_val = np.expand_dims(np.append(corr_coefficients, eigenvalues), axis=0)
        fcorr_feat = np.concatenate((fcorr_feat, fcorr_val), axis=0)  # ,np.reshape(data_win,(1,win_size)),axis=0)

    return fcorr_feat


def eemd_imf(data, win_size, sampl_freq):
    num_imfs = 1
    num_win = data.shape[0]
    emd = emd()
    emd.trials = 2
    num_feat = 4 * (num_imfs + 1)

    imfs_stat = np.zeros((0, num_feat))
    for win_num in range(0, num_win):
        data_win = data[win_num, :]
        eimfs = emd(data_win, max_imf=num_imfs)
        if eimfs.shape[0] == 1:
            eimfs = np.append(eimfs, np.reshape(data_win, (1, win_size)), axis=0)

        imf_mean = np.mean(eimfs, axis=1)
        imf_var = np.var(eimfs, axis=1)
        imf_skew = scipy.stats.skew(eimfs, axis=1)
        imf_mean_freq = weighted_mean_freq(eimfs, win_size, sampl_freq)
        win_feat = np.reshape(np.concatenate((imf_mean, imf_var, imf_skew, imf_mean_freq)), (1, -1))
        imfs_stat = np.concatenate((imfs_stat, win_feat), axis=0)
        # print win_num

    return imfs_stat


def autocorr(data, win_size):  # win_size, sampl_freq
    num_win = data.shape[0]
    seiz_ac = np.zeros((0, win_size))
    for win_num in range(0, num_win):
        data_win = data[win_num, :]
        result = np.correlate(data_win, data_win, mode='full')
        if (win_size % 2) == 0:
            win_ac = result[round(result.size / 2) - 1:][None, :]
        else:
            win_ac = result[round(result.size / 2):][None, :]
        seiz_ac = np.concatenate((seiz_ac, win_ac), axis=0)

    return seiz_ac


def shannon_entropy(data, win_size):
    sh_entropy = np.empty([len(data), 1])
    probabilities = np.empty([len(data), win_size])
    num_bins = win_size

    for i in range(0, len(data), 1):
        counts, bins = np.histogram(data[i, :], bins=num_bins)
        bins = bins[:-1] + (bins[1] - bins[0]) / 2
        probabilities[i, :] = counts / float(counts.sum())
        probabilities[i, :].sum()  # 1.0 , just to check
        # plt.bar(bins, probabilities[i,:], 2)
        # plt.show()
        nonzero_probabilities = np.where(probabilities[i, :] == 0, 1.e-15, probabilities[i, :])
        sh_entropy[i] = scipy.stats.entropy(nonzero_probabilities.transpose())
    return sh_entropy


def ac_coef(x, win_size):
    lag = win_size / 2
    cor_coef = np.corrcoef(np.array([x[0:len(x) - lag], x[lag:len(x)]]))
    return cor_coef


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def feat_array(data, win_size, stride_size, sampling_freq, verbose = False):
    seizures_n = np.unique(data.index.get_level_values(0))
   
    #seiz_feat = np.zeros((number_win, 0))  # np.array([])
    #chan_feat = {}
    
    features = []
    for seizure_id, seizure_block in data.groupby(level=0):
    #for n in seizures_n:
        channel_features = []
        #seizure_block = data.loc[n]
        seizure_block= np.array(seizure_block)
        #print(f'\n seizure: {n}')
        for ch_num in range(0, data.shape[1]):  # for bipolar montage -1 is added
            #print("\n" + ' channel ' + str(ch_num + 1) + "\n")
            # epoched_data = np.reshape(seizure_block[:, ch_num], (-1, win_size))
            epoched_data_tmp = chunk_data(seizure_block[:, ch_num], win_size, stride_size)

            epoched_data = np.round(epoched_data_tmp, 3) # ------- activated for camparison with farroks scripts
            #epoched_data = epoched_data_tmp #### ------ activate for own analysis

            #print('Epoched_data size: ' + str(epoched_data.shape))
            # fft_block ['seizure' + str(sz_num)]= np.fft.rfft(epoched_data, n=None, axis= 2) #######################
            # print 'fft size: ' + str(fft_block['seizure' + str(sz_num)].shape) ####################
            # f, power_block = scipy.signal.periodogram(epoched_data, fs=sampling_freq, window='hann', axis=1, scaling='spectrum')
            f, power_block = scipy.signal.welch(epoched_data, fs=sampling_freq, window='hann', axis=1, scaling='spectrum') #----------------# ERROR source at 1sec window, non-overlaping -> nperseg=None
            fpower_mean = np.mean(power_block, axis=1)[:, None]  # power_block['seizure' + str(sz_num)]
            fpower_var = np.var(power_block, axis=1)[:, None]
            maxpower_inx = np.argmax(power_block, axis=1)[:, None]
            fpower_max = f[maxpower_inx]
            #print('power size: ' + str(power_block.shape))
            #print('power_var size: ' + str(fpower_var.shape))
            #print('power_mean size: ' + str(fpower_mean.shape))
            #print('power_max size: ' + str(fpower_max.shape))

            # EEG frequency bands

            # theta band
            # thetaband_st = int((np.where(f == 4)[0])[0])
            thetaband_st = find_nearest(f, 4)[0]
            # thetaband_end = int((np.where(f == 8)[0])[0])
            thetaband_end = find_nearest(f, 8)[0]
            thetaband_power = power_block[:, thetaband_st:thetaband_end + 1].sum(axis=1)[:, None]
            #print("theta band power size: " + str(thetaband_power.shape))

            # bata band
            # betaband_st = int((np.where(f == 13)[0])[0])
            betaband_st = find_nearest(f, 13)[0]
            # betaband_end = int((np.where(f == 30)[0])[0])
            betaband_end = find_nearest(f, 30)[0]
            betaband_power = power_block[:, betaband_st:betaband_end + 1].sum(axis=1)[:, None]
            #print("beta band power size: " + str(betaband_power.shape))

            # gamma band
            # gammaband_st = int((np.where(f == 30)[0])[0])
            gammaband_st = find_nearest(f, 30)[0]
            # gammaband_end = int((np.where(f == 45)[0])[0])
            gammaband_end = find_nearest(f, 45)[0]
            gammaband_power = power_block[:, gammaband_st:gammaband_end + 1].sum(axis=1)[:, None]
            #print("gamma band power size: " + str(gammaband_power.shape))

            # Epileptogenecity index
            # lowfreq_band_st = int((np.where(f == 3)[0])[0])
            lowfreq_band_st = find_nearest(f, 3)[0]
            # lowfreq_band_end = int((np.where(f == 12)[0])[0])
            lowfreq_band_end = find_nearest(f, 12)[0]
            lowfreq_band_power = power_block[:, lowfreq_band_st:lowfreq_band_end + 1].sum(axis=1)[:, None]
            nonzero_lowfreq_bandpower = np.where(lowfreq_band_power == 0, 1.e-15, lowfreq_band_power)
            #print("low frequency band power size: " + str(nonzero_lowfreq_bandpower.shape))
            epi_index = (betaband_power + gammaband_power) / nonzero_lowfreq_bandpower
            #print('epileptogenecity index size: ' + str(epi_index.shape))

            # time domain features

            # line length
            llength = linelength(epoched_data)[:, None]
            #print('line length size: ' + str(llength.shape))

            # maximum amplitude
            maximum_amp = np.amax(np.absolute(epoched_data), axis=1)[:, None]
            #print('maximum amplitude size: ' + str(maximum_amp.shape))

            # mean value
            mean_val = epoched_data.mean(axis=1)[:, None]
            #print('time domain amplitude mean size: ' + str(mean_val.shape))

            # variance
            var_val = epoched_data.var(axis=1)[:, None]
            #print('time domain amplitude variance size: ' + str(var_val.shape))

            # skewness
            skew_val = scipy.stats.skew(epoched_data, axis=1)[:, None]
            #print('time domain amplitude skewness size: ' + str(skew_val.shape))

            # kurtosis
            kurt_val = scipy.stats.kurtosis(epoched_data, axis=1)[:, None]
            #print('time domain amplitude kurtosis size: ' + str(kurt_val.shape))

            # mean absolute deviation
            mad_val = robust.mad(epoched_data, axis=1)[:, None]
            #print('time domain amplitude mean absolute deviation size: ' + str(mad_val.shape))

            # Autocorrelation
            # ->auto_corr = np.amin(autocorr(epoched_data, win_size), axis=1)[:, None]
            # auto_coef = ac_coef(epoched_data, win_size) #[:, None]
            # ->print("(minimum value)auto-correlation size: " + str(auto_corr.shape))

            # shannon Entropy in time domain

            # plt.hist(epoched_data[2,:], bins=10)
            # plt.show()
            # std_val = epoched_data.std(axis=1)[:, None]
            # probabilities = np.empty([len(epoched_data),win_size])
            # for i in range(1, len(epoched_data), 1):
            #      #values = [value for value in range(int(np.amin(epoched_data[i,:])), int(np.amax(epoched_data[i,:])))]
            #      values = epoched_data[i, :]
            #      dist = norm(mean_val[i], std_val[i])
            #      for k in range(1, win_size, 1):
            #         probabilities[i, k] = dist.pdf(values[k])
            # plt.plot(dist)
            # entropy_val = scipy.stats.entropy(probabilities.transpose(), qk=None, base=None)[:, None]

            # this is the working version:
            # entropy_val = np.empty([len(epoched_data), 1])
            # probabilities = np.empty([len(epoched_data), win_size])
            # num_bins = win_size
            #
            # for i in range(0, len(epoched_data), 1):
            #     counts, bins = np.histogram(epoched_data[i, :], bins=num_bins)
            #     bins = bins[:-1] + (bins[1] - bins[0]) / 2
            #     probabilities[i, :] = counts / float(counts.sum())
            #     print
            #     probabilities[i, :].sum()  # 1.0
            #     #plt.bar(bins, probabilities[i,:], 2)
            #     #plt.show()
            #     entropy_val[i] = scipy.stats.entropy(probabilities[i, :].transpose())

            entropy_val = shannon_entropy(epoched_data, win_size)
            #print('shannon entropy size: ' + str(entropy_val.shape))

            # entropy in frequency domain

            spec_entropy = np.empty([len(epoched_data), 1])
            for j in range(0, len(epoched_data), 1):
                spec_entropy[j] = antropy.spectral_entropy(epoched_data[j, :], sampling_freq, method='welch',
                                                        nperseg=sampling_freq,
                                                        normalize=False)  # sampling_freq is set to 96, because the signal was filtered with high cutoff freq = 48
            #print('spectral entropy size: ' + str(spec_entropy.shape))

            # Ensemble empirical mode decomposition

            # ->stat_imfs = eemd_imf(epoched_data, win_size, sampling_freq)
            # ->print 'IMFs Statistics size: ' + str(stat_imfs.shape)

            # making features list

            # chan_feat = np.concatenate((mean_val, var_val, skew_val, kurt_val, mad_val, llength, auto_corr, betaband_power,
            # gammaband_power, epi_index), axis=1) # org 10 epi_index
            #chan_feat[ch_num] = np.concatenate((var_val, skew_val, kurt_val, mad_val, llength, maximum_amp, entropy_val,  # auto_corr, mean_val
             #                                   thetaband_power, betaband_power, gammaband_power, epi_index, spec_entropy, fpower_max, fpower_mean, fpower_var), axis=1)
            #------#seiz_feat = np.concatenate((seiz_feat, chan_feat[ch_num]), axis=1).astype(np.float32)
            #for i in [var_val, skew_val, kurt_val, mad_val, llength, maximum_amp, entropy_val, thetaband_power, betaband_power, gammaband_power, epi_index, spec_entropy, fpower_max, fpower_mean, fpower_var]:
            #    print(len(i), i.shape, type(i))
            #    print(i)

            if verbose:
                print(f'\n seizure {seizure_id} - ch_{ch_num +1}: ')
                print('Epoched_data size: ' + str(epoched_data.shape))
                print('power size: ' + str(power_block.shape))
                print('power_var size: ' + str(fpower_var.shape))
                print('power_mean size: ' + str(fpower_mean.shape))
                print('power_max size: ' + str(fpower_max.shape))
                print("theta band power size: " + str(thetaband_power.shape))
                print("beta band power size: " + str(betaband_power.shape))
                print("gamma band power size: " + str(gammaband_power.shape))
                print("low frequency band power size: " + str(nonzero_lowfreq_bandpower.shape))
                print('epileptogenecity index size: ' + str(epi_index.shape))
                print('line length size: ' + str(llength.shape))
                print('maximum amplitude size: ' + str(maximum_amp.shape))
                print('time domain amplitude mean size: ' + str(mean_val.shape))
                print('time domain amplitude variance size: ' + str(var_val.shape))
                print('time domain amplitude skewness size: ' + str(skew_val.shape))
                print('time domain amplitude kurtosis size: ' + str(kurt_val.shape))
                print('time domain amplitude mean absolute deviation size: ' + str(mad_val.shape))
                print('shannon entropy size: ' + str(entropy_val.shape))
                print('spectral entropy size: ' + str(spec_entropy.shape))
                

            feat_df = pd.DataFrame({'var_val': var_val.flatten(),
                                    'skew_val': skew_val.flatten(),
                                    'kurt_val': kurt_val.flatten(),
                                    'mad_val': mad_val.flatten(),
                                    'llength': llength.flatten(),
                                    'maximum_amp': maximum_amp.flatten(),
                                    'entropy_val': entropy_val.flatten(), 
                                    'thetaband_power': thetaband_power.flatten(),
                                    'betaband_power': betaband_power.flatten(),
                                    'gammaband_power': gammaband_power.flatten(), 
                                    'epi_index': epi_index.flatten(),
                                    'spec_entropy': spec_entropy.flatten(),
                                    'fpower_max': fpower_max.flatten(),
                                    'fpower_mean': fpower_mean.flatten(),
                                    'fpower_var':fpower_var.flatten()})
            #print([str(ch_num)],list(feat_df.columns))
            multi_col = pd.MultiIndex.from_product([[f'ch_{ch_num +1}'], list(feat_df.columns)])
            #feat_df.rename(columns= multi_col, inplace=True)
            feat_df.columns = multi_col
            #multi_idx = pd.MultiIndex.from_arrays((1 * np.ones_like(feat_df.index), feat_df.index))
            #feat_df = pd.DataFrame(feat_df, index= multi_idx, columns=multi_col)
            channel_features.append(feat_df)
        feat_df = pd.concat(channel_features, axis=1)  
        multi_idx = pd.MultiIndex.from_arrays((seizure_id * np.ones_like(feat_df.index), feat_df.index))  
        feat_df.index = multi_idx
        features.append(feat_df)
    feature_df = pd.concat(features, axis= 0)    
    #return seiz_feat, chan_feat, feat_df
    return feature_df


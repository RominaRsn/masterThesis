import numpy as np

# export cleaned training
def exportcleaned(num_chanClean, sampling_freq, noisy_data, model):
    # create array to reconstruct the sub seizure in original order
    nEpochsForReconstrcuction = 60 * num_chanClean
    noisy_subs = np.zeros((nEpochsForReconstrcuction, 2 * sampling_freq, 1))

    # datascale = (pmaxn - pminn)
    datascale = 1.0
    for c in range(0, num_chanClean):
        for twosec in range(0, 60):
            for ns in range(0, 2 * sampling_freq):
                noisy_subs[twosec + c * 60, ns, 0] = noisy_data[c, ns + twosec * 500] / datascale

    noisy_subs = noisy_subs - noisy_subs.mean(axis=0, keepdims=True)

    sub_reconstructions = model.predict(noisy_subs)

    nChan = num_chanClean
    nSamples = 2 * sampling_freq * 60
    f = open('/Users/matthiasdumpelmann/data/EEG/simultan/subclinical_cleaned_en.msr', 'w')
    f.write("NumberTimeSteps= %d\r\n" % nSamples)
    f.write("NumberPositions= %d\r\n" % nChan)
    f.write("UnitMeas V\n")
    f.write("UnitTime ms\n")
    f.write("TimeSteps        0(%f)\r\n" % (1000.0 / sampling_freq))
    f.write("Labels\n")
    f.write("T2\tT2n\tT2r\n")
    f.write("ValuesTransposed\tsubclinical_cleaned_en.msm\n")
    f.close()

    fb = open('/Users/matthiasdumpelmann/data/EEG/simultan/subclinical_cleaned_en.msm', "wb")

    outdata = np.zeros((nChan, 120 * sampling_freq))

    nr = 0
    for c in range(0, nChan):
        for twosec in range(0, 60):
            nrec = twosec + c * 60
            reconstruction = np.array(sub_reconstructions[nr])
        #    reconstruction = reconstruction - reconstruction.mean(axis=0, keepdims=True)
            nr = nr + 1
            for s in range(0, 2 * sampling_freq):
                outdata[c, twosec * 2 * sampling_freq + s] = reconstruction[s]

    outdata = outdata - outdata.mean(axis=0, keepdims=True)
    stdout = np.std(outdata)

    for s in range(0, nSamples):
        for c in range(0, nChan):
            #value = np.float32(0)
            #if s > 16:
            #    if s < nSamples - 16:
            value = np.float32(outdata[c, s] * datascale)
  #          if value > 10 * stdout:
  #             value = np.float32(0.0)

   #         if value < -10 * stdout:
   #             value = np.float32(0.0)

            fb.write(value)
    fb.close
    return

def exportcleaned_testData(nChan, sampling_freq, test_reconstructions):

    reconstruction = np.array(test_reconstructions[0])
    outdata = np.zeros((nChan, 120 * sampling_freq))
    nr = 0
    datascale = 1.0
    nSamples = 2 * sampling_freq * 60

    for c in range(0, nChan):
        for twosec in range(0, 60):
            nrec = twosec + c * 60;
            reconstruction = np.array(test_reconstructions[nr])
            #     reconstruction = reconstruction - reconstruction.mean(axis=0, keepdims=True)
            nr = nr + 1
            for s in range(0, 2 * sampling_freq):
                outdata[c, twosec * 2 * sampling_freq + s] = reconstruction[s]

    outdata = outdata - outdata.mean(axis=0, keepdims=True)

    f = open('/Users/matthiasdumpelmann/data/EEG/simultan/testdata_reconstructed_e4n.msr', 'w')
    f.write("NumberTimeSteps= %d\r\n" % nSamples);
    f.write("NumberPositions= %d\r\n" % nChan);
    f.write("UnitMeas V\n");
    f.write("UnitTime ms\n");
    f.write("TimeSteps        0(%f)\r\n" % (1000.0 / sampling_freq));
    f.write("Labels\n");
    f.write("T2\tT2n\tT2r\n");
    f.write("ValuesTransposed\ttestdata_reconstructed_e4n.msm\n");
    f.close()

    fb = open('/Users/matthiasdumpelmann/data/EEG/simultan/testdata_reconstructed_e4n.msm', "wb")

    stdout = np.std(outdata)

    for s in range(0, nSamples):
        for c in range(0, nChan):
            value = np.float32(0)
            if s > 16:
                if s < nSamples - 16:
                    value = np.float32(outdata[c, s] * datascale)
            if value > 10 * stdout:
                value = np.float32(0.0)

            if value < -10 * stdout:
                value = np.float32(0.0)

            fb.write(value)
    fb.close

    return

def exportpure_testData(nChan, sampling_freq, pure_input_test):
    outdata = np.zeros((nChan, 120 * sampling_freq))
    nr = 0
    datascale = 1.0
    nSamples = 2 * sampling_freq * 60
    for c in range(0, nChan):
        for twosec in range(0, 60):
            pure_data = np.array(pure_input_test[nr])
            nr = nr + 1
            for s in range(0, 2 * sampling_freq):
                outdata[c, twosec * 2 * sampling_freq + s] = pure_data[s]

    outdata = outdata - outdata.mean(axis=0, keepdims=True)

    f = open('/Users/matthiasdumpelmann/data/EEG/simultan/testdata_pure_e.msr', 'w')
    f.write("NumberTimeSteps= %d\r\n" % nSamples);
    f.write("NumberPositions= %d\r\n" % nChan);
    f.write("UnitMeas V\n");
    f.write("UnitTime ms\n");
    f.write("TimeSteps        0(%f)\r\n" % (1000.0 / sampling_freq));
    f.write("Labels\n");
    f.write("T2\tT2n\tT2r\n");
    f.write("ValuesTransposed\ttestdata_pure_e.msm\n");
    f.close()

    fb = open('/Users/matthiasdumpelmann/data/EEG/simultan/testdata_pure_e.msm', "wb")

    stdout = np.std(outdata)

    for s in range(0, nSamples):
        for c in range(0, nChan):
            value = np.float32(0)
            if s > 8:
                if s < nSamples - 16:
                    value = np.float32(outdata[c, s] * datascale)
            if value > 10 * stdout:
                value = np.float32(0.0)

            if value < -10 * stdout:
                value = np.float32(0.0)

            fb.write(value)
    fb.close()
    return

def exportnoisy_testData(nChan, sampling_freq, noisy_input_test):
    outdata = np.zeros((nChan, 120 * sampling_freq))
    datascale = 1.0
    nSamples = 2 * sampling_freq * 60
    nr = 0
    for c in range(0, nChan):
        for twosec in range(0, 60):
            noisy_data = np.array(noisy_input_test[nr])
            nr = nr + 1
            for s in range(0, 2 * sampling_freq):
                outdata[c, twosec * 2 * sampling_freq + s] = noisy_data[s]

    outdata = outdata - outdata.mean(axis=0, keepdims=True)

    f = open('/Users/matthiasdumpelmann/data/EEG/simultan/testdata_noisy_e4n.msr', 'w')
    f.write("NumberTimeSteps= %d\r\n" % nSamples);
    f.write("NumberPositions= %d\r\n" % nChan);
    f.write("UnitMeas V\n");
    f.write("UnitTime ms\n");
    f.write("TimeSteps        0(%f)\r\n" % (1000.0 / sampling_freq));
    f.write("Labels\n");
    f.write("T2\tT2n\tT2r\n");
    f.write("ValuesTransposed\ttestdata_noisy_e4n.msm\n");
    f.close()

    fb = open('/Users/matthiasdumpelmann/data/EEG/simultan/testdata_noisy_e4n.msm', "wb")

    stdout = np.std(outdata)

    for s in range(0, nSamples):
        for c in range(0, nChan):
            value = np.float32(0)
            if s > 8:
                if s < nSamples - 16:
                    value = np.float32(outdata[c, s] * datascale)
            if value > 10 * stdout:
                value = np.float32(0.0)

            if value < -10 * stdout:
                value = np.float32(0.0)

            fb.write(value)
    fb.close()
    return

def exportfiltered_testData(nChan, sampling_freq, filteredsignal):
    filteredshape = filteredsignal.shape
    nFilterdSamples = filteredshape[0]
    datascale = 1.0
    f = open('/Users/matthiasdumpelmann/data/EEG/simultan/testdata_filtered_e.msr', 'w')
    f.write("NumberTimeSteps= %d\r\n" % nFilterdSamples);
    f.write("NumberPositions= %d\r\n" % 1);
    f.write("UnitMeas V\n");
    f.write("UnitTime ms\n");
    f.write("TimeSteps        0(%f)\r\n" % (1000.0 / sampling_freq));
    f.write("Labels\n");
    f.write("LP\n");
    f.write("ValuesTransposed\ttestdata_filtered_e.msm\n");
    f.close()

    fb = open('/Users/matthiasdumpelmann/data/EEG/simultan/testdata_filtered_e.msm', "wb")

    for s in range(0, nFilterdSamples):
        value = np.float32(0)
        value = np.float32(filteredsignal[s] * datascale)
        fb.write(value)
    fb.close
    return

def export_cleanedUNEEGData(sampling_freq, cleaned_data):

    datascale = 1.0
    nChan, nSamples = cleaned_data.shape

    outdata = np.zeros((nChan, nSamples))

    for c in range(0, nChan):
        for s in range(0, nSamples):
            outdata[c, s] = cleaned_data[c,s]

    outdata = outdata - outdata.mean(axis=0, keepdims=True)

    f = open('/Users/matthiasdumpelmann/data/EEG/simultan/cleaned_uneegsxxn.msr', 'w')
    f.write("NumberTimeSteps= %d\r\n" % nSamples);
    f.write("NumberPositions= %d\r\n" % nChan);
    f.write("UnitMeas V\n");
    f.write("UnitTime ms\n");
    f.write("TimeSteps        0(%f)\r\n" % (1000.0 / sampling_freq));
    f.write("Labels\n");
    f.write("SQ_D\tSQ_P\tT2r\n");
    f.write("ValuesTransposed\tcleaned_uneegsxxn.msm\n");
    f.close()

    fb = open('/Users/matthiasdumpelmann/data/EEG/simultan/cleaned_uneegsxxn.msm', "wb")

    stdout = np.std(outdata)

    for s in range(0, nSamples):
        for c in range(0, nChan):
            value = np.float32(0)
            if s > 8:
                if s < nSamples - 16:
                    value = np.float32(outdata[c, s] * datascale)
            if value > 10 * stdout:
                value = np.float32(0.0)

            if value < -10 * stdout:
                value = np.float32(0.0)

            fb.write(value)
    fb.close()
    return

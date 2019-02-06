import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from recordclass import recordclass


def calcStepStats(chemIx, predictions, labels):
    muList = np.unique(labels[chemIx])
    nSteps = muList.size
    nChems = np.shape(labels)[1]
    # Initialize stats structure
    stats = recordclass('stats', 'labels, predRmse, predSnr, predSnre, mean, sd, n, sem')
    np.seterr(divide='ignore', invalid='ignore')  # SNR and SNRE calculations divide by 0
    # initialize variables for calculating stats
    signal = np.squeeze(predictions, axis=2)
    truth = np.array(labels)
    noise = np.array(signal - truth)
    estimate = np.empty(signal.shape)
    noiseEst = np.empty(signal.shape)
    stats.labels = np.empty((nSteps, nChems))
    stats.prediction_RMSE = np.empty((nSteps, nChems))
    stats.prediction_SNR = np.empty((nSteps, nChems))
    stats.prediction_SNRE = np.empty((nSteps, nChems))
    stats.mean = np.empty((nSteps, nChems))
    stats.sd = np.empty((nSteps, nChems))
    stats.n = np.empty((nSteps, nChems))
    stats.sem = np.empty((nSteps, nChems))

    stats.fullRmse = np.empty(nChems)
    stats.fullSnr = np.empty(nChems)
    stats.fullSnre = np.empty(nChems)
    # Calculate stats for each step
    for ix in range(nSteps):
        selectIx = np.where(labels[chemIx] == muList[ix])
        estimate[selectIx] = signal[selectIx, :].mean(axis=1)
        noiseEst[selectIx] = signal[selectIx] - estimate[selectIx]
        ssSig = (signal[selectIx, :] ** 2).sum(axis=1)  # sum square signal
        ssNoise = (noise[selectIx, :] ** 2).sum(axis=1)  # sum square noise
        ssNoiseEst = (noiseEst[selectIx, :] ** 2).sum(axis=1)  # sum square noise Estimate
        stats.labels[ix, :] = truth[selectIx[0][0], :]

        stats.prediction_RMSE[ix, :] = np.sqrt(np.square(noise[selectIx, :]).mean(axis=1))
        stats.prediction_SNR[ix, :] = 10 * np.log10((ssSig - ssNoise) / ssNoise)  # formula for SNR
        stats.prediction_SNRE[ix, :] = 10 * np.log10((ssSig - ssNoiseEst) / ssNoiseEst)
        stats.mean[ix, :] = np.mean(signal[selectIx, :])
        stats.sd[ix, :] = np.std(signal[selectIx, :])
        stats.n[ix, :] = np.size(signal[selectIx, :])
        stats.sem[ix, :] = stats.sd[ix, :] / np.sqrt(stats.n[ix, :])
        # Calculate full data statistics
        stats.fullRmse = np.sqrt((noise ** 2).mean(axis=0))
        ssSignal = np.square(signal).sum(axis=0)
        ssNoise = np.square(noise).sum(axis=0)
        ssNoiseEst = np.square(noiseEst).sum()
        stats.fullSnr = 10 * np.log10((ssSignal - ssNoise) / ssNoise)
        stats.fullSnre = 10 * np.log10((ssSignal - ssNoiseEst) / ssNoiseEst)
        np.seterr(divide=None, invalid=None)  # revert to warning for division by 0
    return stats


def plot_Calibration(time, predictions, labels, targetAnalyte, chemIx, stats):
    X = time
    Y = predictions
    L = np.array(labels)
    chemLabel = targetAnalyte[chemIx]
    labColor = 'k'
    units = ''
    if chemLabel == 'NE':
        chemLabel = 'NE'
        units = '(nM)'
        labColor = 'm'
    if (chemLabel == 'Dopamine') | (chemLabel == 'DA'):
        chemLabel = 'DA'
        units = '(nM)'
        labColor = 'c'
    if (chemLabel == 'Serotonin') | (chemLabel == '5HT'):
        chemLabel = '5HT'
        units = '(nM)'
        labColor = 'y'
    if chemLabel == '5HIAA':
        chemLabel = '5HIAA'
        units = '(nM)'
        labColor = 'g'
    if chemLabel == 'pH':
        chemLabel = 'pH'
        units = ''
        labColor = 'k'

    muLabel = ''.join([chemLabel, units])
    gs = GridSpec(7, 5)
    # Plot Predictions
    ax1 = plt.subplot(gs[1:4, :])
    hPred = plt.scatter(X, Y[:, chemIx], marker='.', color=labColor)
    plt.title(chemLabel)
    plt.xlabel('samples')
    plt.ylabel(muLabel)

    # Plot actual concentrations
    hAct = plt.scatter(X, L[:, chemIx], color='red', marker='.', linewidth=0.5)
    ax1.legend((hPred, hAct), ('predicted', 'actual'))
    plt.axis('tight')

    # Plot RMSE
    ax2 = plt.subplot(gs[5:7, 0:2])
    # for chemIx in range(nChems):
    y = stats.predRmse[:, chemIx]
    x = stats.labels[:, chemIx]
    ax2.scatter(x, y, color=labColor)
    ax2.plot(plt.xlim(), [stats.fullRmse[chemIx], stats.fullRmse[chemIx]], linestyle='--', markersize=1,
             color=labColor)
    plt.title('RMSE')
    plt.xlabel(muLabel)
    plt.ylabel(''.join(['RMSE', units]))
    plt.grid()
    plt.axis('tight')

    # Plot SNR
    ax3 = plt.subplot(gs[5:7, 3:5])
    y = stats.predSnr[:, chemIx]
    x = stats.labels[:, chemIx]
    ax3.scatter(x, y, color=labColor)
    ax3.plot(plt.xlim(), [stats.fullSnr[chemIx], stats.fullSnr[chemIx]], linestyle='--', markersize=1, color=labColor)
    plt.title('SNR')
    plt.xlabel(muLabel)
    plt.ylabel('SNR (dB)')
    plt.grid()
    plt.axis('tight')
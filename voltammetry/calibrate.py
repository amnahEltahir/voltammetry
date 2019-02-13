import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from recordclass import recordclass
import scipy.io
import random as rand
import time
from glmnet_python import cvglmnet, cvglmnetPredict


def best_alpha(training, nAlphas=11, family='mgaussian', ptype='mse', nfolds=10, parallel=True, keep=False,
               grouped=True, random_seed=0, fnY=lambda x: np.diff(x)):
    """
    Find best alpha based on minimum cross-validated error.
    :param training: voltammogram data structure
    :param nAlphas: int, number of alphas to test between 0 and 1 (default = 11)
    :param family: string, cvglment "family" option (default="mgaussian", mixed gaussian)
    :param ptype: string, penalty type
    :param nfolds: float, number of cross validation folds (default = 10)
    :param parallel: boolean, use multiple cores for training (default = True)
    :param keep: oolean, cvglmnet parameter (default = False)
    :param grouped: boolean, cvglmnet parameter (default = True)
    :param random_seed: integer, specifies random seed (default = 0)
    :param fnY: function applied to voltammogram (default = lambda x: np.diff(x), first derivative)
    :return: bestAlpha: float, optimum alpha based on cv error
    """
    alphaRange = np.linspace(0, 1, nAlphas)
    cvFitList = [None] * nAlphas
    t = time.time()

    for i in range(nAlphas):
        alpha = alphaRange[i]
        X = fnY(training.vgrams).astype(float)
        Y = np.array(training.labels).astype(float)
        rand.seed(random_seed)
        foldid = scipy.random.choice(nfolds, training.vgrams.shape[0], replace=True)
        cvFitList[i] = cvglmnet(x=X, y=Y, family=family, alpha=alpha, ptype=ptype, nfolds=nfolds, foldid=foldid,
                                parallel=parallel, keep=keep, grouped=grouped)
    elapsed = time.time() - t
    print('TRAINING COMPLETE', '{:.3f}'.format(elapsed), ' seconds. \n')
    # collect mean cross-validated error for each fit
    cvm = pd.Series(np.empty(nAlphas))

    for i in range(nAlphas):
        fit = cvFitList[i]
        cvm[i] = fit['cvm'][fit['lambdau'] == fit['lambda_min']]

    bestCvm = cvm.min()
    bestAlphaIx = cvm.idxmin()
    bestAlpha = alphaRange[bestAlphaIx]
    print('Best alpha = ', '{:.1f}'.format(bestAlpha), ' (error = ', '{:.2f}'.format(bestCvm), ')\n')
    return bestAlpha


def train_analyte(training, family='mgaussian', alpha=1, ptype='mse', nfolds=10, parallel=True, keep=False,
                  grouped=True, random_seed=0, fnY=lambda x: np.diff(x)):
    """
    Cross validation training to generate elastic net model.
    :param training: Voltamogram_data structure with training data 
    :param family: string, cvglment "family" option (default="mgaussian", mixed gaussian)
    :param alpha: float [0,1] for elastic net (default = 1, LASSO)
    :param ptype: string, penalty type
    :param nfolds: double, number of cross validation folds (default = 10)
    :param parallel: boolean, use multiple cores for training (default = True)
    :param keep: boolean, cvglmnet parameter (default = False)
    :param grouped: boolean, cvglmnet parameter (default = True)
    :param random_seed: integer, specifies random seed (default = 0)
    :param fnY: function applied to voltammogram (default = lambda x: np.diff(x), first derivative)
    :return: cvFit: cvfit object, model based on training data
    """
    rand.seed(random_seed)
    foldid = scipy.random.choice(nfolds, training.vgrams.shape[0], replace=True)
    x = fnY(training.vgrams).astype(float)
    y = np.array(training.labels).astype(float)
    t = time.time()
    cvFit = cvglmnet(x=x, y=y, family=family, alpha=alpha, ptype=ptype, nfolds=nfolds, foldid=foldid, parallel=parallel,
                     keep=keep, grouped=grouped)
    elapsed = time.time() - t
    print('TRAINING COMPLETE ', '{:.3f}'.format(elapsed), ' seconds. \n')
    return cvFit


def test_analyte(testing, cvFit, fnY=lambda x: np.diff(x), s='lambda_min'):
    """
    Test elastic net model
    :param testing: Voltammogram data structure with testing data
    :param cvFit: cvFit object, fit calculated using training_analyte
    :param fnY: function applied to voltammogram (default = lambda x: np.diff(x), first derivative)
    :param s: int, select lambda based on MSE (default = 'lambda_min', lambda of minimum MSE)
    :return: yy: array, predictions based on cvFit
    """
    xx = fnY(testing.vgrams)
    yy = cvglmnetPredict(cvFit, xx, s)
    return yy


def calcStepStats(chemIx, predictions, labels):
    """
    Calculate statistics of model predictions
    :param chemIx: int, index of target analyte
    :param predictions: ndarray, predictions from testing
    :param labels: Data Frame, all test labels
    :return: stats: stats structure, statistics calculated on predictions
    """
    muList = np.unique(labels[:, chemIx])
    nSteps = muList.size
    nChems = np.shape(labels)[1]
    # Initialize stats structure
    stats = recordclass('stats', 'labels, prediction_RMSE, prediction_SNR, prediction_SNRE, mean, sd, n, sem')
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

    stats.fullRMSE = np.empty(nChems)
    stats.fullSNR = np.empty(nChems)
    stats.fullSNRE = np.empty(nChems)
    # Calculate stats for each step
    for ix in range(nSteps):
        selectIx = np.where(labels[:, chemIx] == muList[ix])[0]
        stepSize = len(selectIx)
        estimate[selectIx, :] = np.tile(signal[selectIx, :].mean(axis=0), (stepSize, 1))
        noiseEst[selectIx, :] = signal[selectIx] - estimate[selectIx]
        stats.labels[ix, :] = truth[selectIx[0], :]

        stats.prediction_RMSE[ix, :] = np.sqrt(np.square(noise[selectIx, :]).mean(axis=0))
        stats.prediction_SNR[ix, :] = calculate_SNR(signal[selectIx, :], noise[selectIx, :])
        stats.prediction_SNRE[ix, :] = calculate_SNR(signal[selectIx, :], noiseEst[selectIx, :])
        stats.mean[ix, :] = np.mean(signal[selectIx, :])
        stats.sd[ix, :] = np.std(signal[selectIx, :])
        stats.n[ix, :] = np.size(signal[selectIx, :])
        stats.sem[ix, :] = stats.sd[ix, :] / np.sqrt(stats.n[ix, :])
    # Calculate full data statistics
    stats.fullRMSE = np.sqrt((noise ** 2).mean(axis=0))
    stats.fullSNR = calculate_SNR(signal, noise)
    stats.fullSNRE = calculate_SNR(signal, noiseEst)
    np.seterr(divide=None, invalid=None)  # revert to warning for division by 0
    return stats


def plot_Calibration(time_array, predictions, labels, targetAnalyte, chemIx, stats):
    """
    Plot fits with labels, as well as RMSE and SNR for a given analyte.
    :param time_array: array, time used in x axis
    :param predictions: array, predictions from cvglmnetPredict
    :param labels: array, label concentrations
    :param targetAnalyte: str array, list of analytes
    :param chemIx: int, index of chemical being modeled
    :param stats: stats structure, structure of calculated statistics for variable
    :return: fig: handle for figure of results
    """
    X = time_array
    Y = predictions
    L = np.array(labels)
    chemLabel = targetAnalyte[chemIx]
    labColor = 'k'
    units = ''
    if chemLabel == 'NE':
        chemLabel = 'NE'
        units = '(nM)'
        labColor = '#b4531f'
    if (chemLabel == 'Dopamine') | (chemLabel == 'DA'):
        chemLabel = 'DA'
        units = '(nM)'
        labColor = '#1f77b4'
    if (chemLabel == 'Serotonin') | (chemLabel == '5HT'):
        chemLabel = '5HT'
        units = '(nM)'
        labColor = '#b49e1f'
    if chemLabel == '5HIAA':
        chemLabel = '5HIAA'
        units = '(nM)'
        labColor = '#871fb4'
    if chemLabel == 'pH':
        chemLabel = 'pH'
        units = ''
        labColor = '#3ebd30'
    fig = plt.figure()
    muLabel = ''.join([chemLabel, units])
    gs = GridSpec(7, 5)
    # Plot Predictions
    ax1 = plt.subplot(gs[1:4, :])
    hPred = plt.scatter(X, Y[:, chemIx], marker='.', color=labColor)
    plt.title(chemLabel)
    plt.xlabel('samples')
    plt.ylabel(muLabel)

    # Plot actual concentrations
    hAct = plt.scatter(X, L[:, chemIx], color='k', marker='.', linewidth=0.5)
    ax1.legend((hPred, hAct), ('predicted', 'actual'))
    plt.axis('tight')

    # Plot RMSE
    ax2 = plt.subplot(gs[5:7, 0:2])
    # for chemIx in range(nChems):
    y = stats.prediction_RMSE[:, chemIx]
    x = stats.labels[:, chemIx]
    ax2.scatter(x, y, color=labColor)
    ax2.plot(plt.xlim(), [stats.fullRMSE[chemIx], stats.fullRMSE[chemIx]], linestyle='--', markersize=1,
             color=labColor)
    plt.title('RMSE')
    plt.xlabel(muLabel)
    plt.ylabel(''.join(['RMSE', units]))
    plt.grid()
    plt.axis('tight')

    # Plot SNR
    ax3 = plt.subplot(gs[5:7, 3:5])
    y = stats.prediction_SNR[:, chemIx]
    x = stats.labels[:, chemIx]
    ax3.scatter(x, y, color=labColor)
    ax3.plot(plt.xlim(), [stats.fullSNR[chemIx], stats.fullSNR[chemIx]], linestyle='--', markersize=1, color=labColor)
    plt.title('SNR')
    plt.xlabel(muLabel)
    plt.ylabel('SNR (dB)')
    plt.grid()
    plt.axis('tight')

    return fig


def calculate_SNR(sig, noise):
    """
    Calculate SNR in decibels
    :param sig: array of signal
    :param noise: array of noise
    :return: array, SNR calculated as 10*log10(Power_signal-Power_noise/Power_noise)
    """
    P_sig = (sig ** 2).sum(axis=0) / len(sig)  # sum square signal
    P_noise = (noise ** 2).sum(axis=0) / len(sig)
    SNR = 10 * np.log10((P_sig - P_noise) / P_noise)
    return SNR

import pandas as pd
import math
import numpy as np
import random as rand
from recordclass import recordclass
from statsmodels import robust


def find_stable_section(Voltammogram, window_size):
    """
    Find portion of calibration that is most stable.
    """
    samplewise = 0
    sweepwise = 1
    num_experiments = np.shape(Voltammogram)[2]
    sample_num = np.shape(Voltammogram)[samplewise]
    good_window = np.zeros((sample_num, window_size, num_experiments))
    exclude_ix = []

    for i in range(num_experiments):
        vgrams = Voltammogram[:, :, i]
        n_sweeps = np.shape(vgrams)[sweepwise]

        # Calculate "before" and "after" window sizes based on move
        winHead = math.floor(window_size / 2)
        winTail = math.ceil(window_size / 2)

        # Step 1: Find median waveform for the window centered at each step
        vDF = pd.DataFrame(vgrams)
        M = vDF.rolling(window_size).median()
        # Step 2: Find the "difference" by subtracting the window median from each sweep
        D = vDF - M
        # Step 3: Find the RMS (samplewise) of the difference
        r = (D ** 2).mean(axis=samplewise) ** 0.5
        # Step 4: Find the mean RMS of the window centered on each sweep
        q = r.rolling(window_size).mean()
        # Step 5: Find the window centered on the sweep with the lowest q value in the
        # second half of the data
        halfIx = math.floor(n_sweeps / 2)
        startIx = halfIx + winHead
        endIx = n_sweeps - winTail - 1
        bestWinCenterIx = int(np.argmin(q[endIx:startIx:-1]))
        gw = np.array(range(bestWinCenterIx - winHead, bestWinCenterIx + winTail))
        good_window[:, :, i] = vgrams[:, gw]
        # Step 6: mark any sweeps where the RMS of the difference is an outlier
        ex = madOutlier(r.loc[gw])
        exclude_ix.append(ex)
        #  TODO: Exclude the outliers from use in analysis
    return [good_window, exclude_ix]


def partitionData(voltammograms, labels, trainingSampleSize):
    rand.seed(0)  # random sampling reproducible
    N = voltammograms.shape[2]  # Number of concentrations
    Rn = voltammograms.shape[1]  # Sweep number
    Sn = voltammograms.shape[0]  # Sample number
    # chems    = len(labels.targetAnalyte)  # Number of chemicals analyzed
    # initialize training and testing structures
    training = recordclass('training', 'sampleSize, index, vgrams, labels, experiment')
    testing = recordclass('testing', 'sampleSize, index, vgrams, labels, experiment')
    ## Partition each experiment
    # training partition
    training.sampleSize = trainingSampleSize
    training.index = np.zeros((Rn, N))
    training.vgrams = []
    training.labels = []
    training.experiment = np.zeros((trainingSampleSize, N))
    # testing partition
    testingSampleSize = Rn - trainingSampleSize
    testing.sampleSize = testingSampleSize
    testing.index = np.zeros((Rn, N))
    testing.vgrams = []
    testing.labels = []
    testing.experiment = np.zeros((testingSampleSize, N))
    # Build training and testing structures
    for i in range(N):
        vgrams = pd.DataFrame(voltammograms[:, :, i])
        labs = np.array(labels)[i]
        population = range(Rn)
        sample = rand.sample(population, training.sampleSize)
        index = []
        for j in population: index.append(j in sample)
        # assign training data
        training.index[:, i] = np.array(index)
        training.vgrams.append(vgrams.loc[:, index])
        training.labels.append(pd.DataFrame(pd.np.tile(labs, (trainingSampleSize, 1))))
        training.experiment[:, i] = [N + 1] * trainingSampleSize
        # assign testing data
        testing.index[:, i] = ~np.array(index)
        testing.vgrams.append(vgrams.loc[:, ~np.array(index)])
        testing.labels.append(pd.DataFrame(pd.np.tile(labs, (testingSampleSize, 1))))
        testing.experiment[:, i] = [N + 1] * testingSampleSize

    return [training, testing]


def flattenData(training, testing):
    training.index = np.where(np.hstack((training.index)))
    training.vgrams = pd.concat(training.vgrams, axis=1)
    training.labels = pd.concat(training.labels, axis=0)
    training.experiment = np.vstack(training.experiment)
    training.n = np.shape(training.index)

    training.vgrams = np.transpose(training.vgrams)

    testing.index = np.where(np.hstack((testing.index)))
    testing.vgrams = pd.concat(testing.vgrams, axis=1)
    testing.labels = pd.concat(testing.labels, axis=0)
    testing.experiment = np.vstack(testing.experiment)
    testing.n = np.shape(testing.index)

    testing.vgrams = np.transpose(testing.vgrams)
    return [training, testing]


def madOutlier(data,thresh=3.5):

    # use Median Absolute Deviation method, default in MATLAB
    MAD = robust.mad(data[np.isfinite(data)])
    excludeIx = np.where((data < data.median() - thresh*MAD) | (data > data.median() + 3.5*MAD))
    return excludeIx

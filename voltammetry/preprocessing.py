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
    sample_wise = 0
    sweep_wise = 1
    num_experiments = np.shape(Voltammogram)[2]
    sample_num = np.shape(Voltammogram)[sample_wise]
    good_window = np.zeros((sample_num, window_size, num_experiments))
    exclude_ix = []

    for i in range(num_experiments):
        vgrams = Voltammogram[:, :, i]
        n_sweeps = np.shape(vgrams)[sweep_wise]

        # Calculate "before" and "after" window sizes based on move
        window_head = math.floor(window_size / 2)
        window_tail = math.ceil(window_size / 2)

        # Step 1: Find median waveform for the window centered at each step
        vgram_df = pd.DataFrame(vgrams)
        vgram_median = vgram_df.rolling(window_size).median()
        # Step 2: Find the "difference" by subtracting the window median from each sweep
        diff_df_median = vgram_df - vgram_median
        # Step 3: Find the RMS (sample_wise) of the difference
        r = (diff_df_median ** 2).mean(axis=sample_wise) ** 0.5
        # Step 4: Find the mean RMS of the window centered on each sweep
        q = r.rolling(window_size).mean()
        # Step 5: Find the window centered on the sweep with the lowest q value in the
        # second half of the data
        half_index = math.floor(n_sweeps / 2)
        start_index = half_index + window_head
        end_index = n_sweeps - window_tail - 1
        best_win_center_index = int(np.argmin(q[end_index:start_index:-1]))
        gw = np.array(range(best_win_center_index - window_head, best_win_center_index + window_tail))
        good_window[:, :, i] = vgrams[:, gw]
        # Step 6: mark any sweeps where the RMS of the difference is an outlier
        ex = mad_outlier(r.loc[gw])
        exclude_ix.append(ex)
        #  TODO: Exclude the outliers from use in analysis
    return [good_window, exclude_ix]


def partition_data(voltammograms, labels, trainingSampleSize):
    rand.seed(0)  # random sampling reproducible
    num_experiments = voltammograms.shape[2]  # Number of concentrations
    num_sweeps = voltammograms.shape[1]  # Sweep number
    # num_samples = voltammograms.shape[0]  # Sample number
    # initialize training and testing structures
    training = recordclass('training', 'sampleSize, index, vgrams, labels, experiment')
    testing = recordclass('testing', 'sampleSize, index, vgrams, labels, experiment')
    # Partition each experiment
    # training partition
    training.sampleSize = trainingSampleSize
    training.index = np.zeros((num_sweeps, num_experiments))
    training.vgrams = []
    training.labels = []
    training.experiment = np.zeros((trainingSampleSize, num_experiments))
    # testing partition
    testing_sample_size = num_sweeps - trainingSampleSize
    testing.sampleSize = testing_sample_size
    testing.index = np.zeros((num_sweeps, num_experiments))
    testing.vgrams = []
    testing.labels = []
    testing.experiment = np.zeros((testing_sample_size, num_experiments))
    # Build training and testing structures
    for i in range(num_experiments):
        vgrams = pd.DataFrame(voltammograms[:, :, i])
        labs = np.array(labels)[i]
        population = range(num_sweeps)
        sample = rand.sample(population, training.sampleSize)
        index = []
        for j in population:
            index.append(j in sample)
        # assign training data
        training.index[:, i] = np.array(index)
        training.vgrams.append(vgrams.loc[:, index])
        training.labels.append(pd.DataFrame(pd.np.tile(labs, (trainingSampleSize, 1))))
        training.experiment[:, i] = [num_experiments + 1] * trainingSampleSize
        # assign testing data
        testing.index[:, i] = ~np.array(index)
        testing.vgrams.append(vgrams.loc[:, ~np.array(index)])
        testing.labels.append(pd.DataFrame(pd.np.tile(labs, (testing_sample_size, 1))))
        testing.experiment[:, i] = [num_experiments + 1] * testing_sample_size

    return [training, testing]


def flatten_data(training, testing):
    training.index = np.where(np.hstack(training.index))
    training.vgrams = pd.concat(training.vgrams, axis=1)
    training.labels = pd.concat(training.labels, axis=0)
    training.experiment = np.vstack(training.experiment)
    training.n = np.shape(training.index)

    training.vgrams = np.transpose(training.vgrams)

    testing.index = np.where(np.hstack(testing.index))
    testing.vgrams = pd.concat(testing.vgrams, axis=1)
    testing.labels = pd.concat(testing.labels, axis=0)
    testing.experiment = np.vstack(testing.experiment)
    testing.n = np.shape(testing.index)

    testing.vgrams = np.transpose(testing.vgrams)
    return [training, testing]


def mad_outlier(data, thresh=3.5):
    # use Median Absolute Deviation method, default in MATLAB
    MAD = robust.mad(data[np.isfinite(data)])
    exclude_index = np.where((data < data.median() - thresh * MAD) | (data > data.median() + 3.5 * MAD))
    return exclude_index

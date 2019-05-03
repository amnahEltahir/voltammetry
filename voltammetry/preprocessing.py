import pandas as pd
import math
import numpy as np
import random as rand
from recordclass import recordclass
from statsmodels import robust


class PreprocessedData:
    def __init__(self, voltammogram_data, muLabels, window_size=1500, trainingSampleSize=500):
        print("Finding stable section with window size", window_size)
        [good_window, exclude_ix] = find_stable_section(voltammogram_data, window_size)
        print("Partitioning data with training sample size", trainingSampleSize)
        [training_part, testing_part] = partition_data(voltammogram_data, muLabels.labels, good_window, exclude_ix,
                                                       trainingSampleSize)
        print("Flattening Data")
        [self.training, self.testing] = flatten_data(training_part, testing_part)
        print("PRE-PROCESSING COMPLETE!!!!")


def find_stable_section(Voltammogram, window_size=150):
    """
    Select window with stable median value
    :param Voltammogram: array, Voltammetry data
    :param window_size: int, number of sweeps in window (default = 150)
    :return: good_window: array of good window index
    :return: exclude_ix: list, indices of outliers
    """
    sample_wise = 0
    sweep_wise = 1
    num_experiments = np.shape(Voltammogram)[2]
    # sample_num = np.shape(Voltammogram)[sample_wise]
    good_window = np.zeros((window_size, num_experiments))
    exclude_ix = []
    for i in range(num_experiments):
        vgrams = Voltammogram[:, :, i]
        n_sweeps = np.shape(vgrams)[sweep_wise]

        # Calculate "before" and "after" window sizes based on move
        window_head = math.floor(window_size / 2)
        window_tail = math.ceil(window_size / 2)
        # Step 1: Find median waveform for the window centered at each step
        vgram_df = pd.DataFrame(vgrams)
        vgram_median = vgram_df.rolling(window_size, center=True, axis=1).median()
        # Step 2: Find the "difference" by subtracting the window median from each sweep
        diff_df_median = vgram_df - vgram_median
        # Step 3: Find the RMS (sample_wise) of the difference
        r = (diff_df_median ** 2).mean(axis=sample_wise) ** 0.5
        # Step 4: Find the mean RMS of the window centered on each sweep
        q = r.rolling(window_size).mean()
        # Step 5: Find the window centered on the sweep with the lowest q value in the
        half_index = math.floor(n_sweeps / 2)
        start_index = half_index + window_head
        end_index = n_sweeps - window_tail - 1
        best_win_center = int(pd.Series.idxmin(q[end_index:start_index:-1]))
        good_window[:, i] = np.array(range(best_win_center - window_head, best_win_center + window_tail))
        # Step 6: mark any sweeps where the RMS of the difference is an outlier
        ex = mad_outlier(r.loc[good_window[:, i]])
        exclude_ix.append(ex)
    good_window = good_window.astype(int)
    return [good_window, exclude_ix]


def partition_data(voltammograms, labels, good_window, exclude_ix, trainingSampleSize=50):
    """
    Partition data into "training" and testing
    :param voltammograms: array of voltammetry data
    :param labels: Data frame of experiment labels
    :param good_window: array, window of region with stable median value
    :param exclude_ix: list, indices of outliers
    :param trainingSampleSize: int, number of sweeps in training
    :return: training: structure with training data
    :return: testing: structure with testing data
    """
    rand.seed(0)  # random sampling reproducible
    num_experiments = voltammograms.shape[2]  # Number of concentrations
    # num_samplePoints = voltammograms.shape[0]  # Number in points per voltammogram
    num_sweeps = good_window.shape[0]  # Number of sweeps in window
    # num_Chems = labels.shape[1]  # Number of chemicals/columns in label variable
    # initialize training and testing structures
    training = recordclass('training', 'sampleSize, index, vgrams, labels, experiment')
    testing = recordclass('testing', 'sampleSize, index, vgrams, labels, experiment')
    # Partition each experiment
    training.sampleSize = trainingSampleSize
    testing_sample_size = num_sweeps - trainingSampleSize
    testing.sampleSize = testing_sample_size
    # training partition
    training.index = [None] * num_experiments
    training.vgrams = [None] * num_experiments
    training.labels = [None] * num_experiments
    training.experiment = [None] * num_experiments
    # testing partition
    testing.index = [None] * num_experiments
    testing.vgrams = [None] * num_experiments
    testing.labels = [None] * num_experiments
    testing.experiment = [None] * num_experiments
    # Build training and testing structures
    for i in range(num_experiments):
        vgrams = pd.DataFrame(voltammograms[:, good_window[:, i], i])
        labs = np.array(labels)[i]
        pop = range(num_sweeps)
        population = list(np.delete(pop, exclude_ix[i]))
        sample = rand.sample(population, training.sampleSize)
        index = []
        for j in population:
            index.append(j in sample)
        # assign training data
        training_index = np.where(index)
        training.index[i] = np.array(training_index[0])
        training.vgrams[i] = vgrams.loc[:, training_index[0]]
        training.labels[i] = np.tile(labs, (len(training_index[0]), 1))
        training.experiment[i] = [num_experiments + 1] * trainingSampleSize
        # assign testing data
        testing_index = np.where(~np.array(index))
        testing.index[i] = np.array(testing_index[0])
        testing.vgrams[i] = vgrams.loc[:, testing_index[0]]
        testing.labels[i] = np.tile(labs, (len(testing_index[0]), 1))
        testing.experiment[i] = [num_experiments + 1] * testing_sample_size

    return [training, testing]


def flatten_data(training, testing):
    """
    Transform voltammogram and label data into proper dimensions for cvglmnet
    :param training: structure containing training and testing data
    :param testing: structure containing testing data
    :return: training: structure with flattened training data
    :return: testing: structure with flattened testing data
    """
    training.index = np.where(np.hstack(training.index) > -1)[0]
    training.vgrams = np.hstack(training.vgrams).transpose()
    training.labels = np.vstack(training.labels)
    training.experiment = np.vstack(training.experiment)
    training.n = np.shape(training.index)

    testing.index = np.where(np.hstack(testing.index) > -1)[0]
    testing.vgrams = np.hstack(testing.vgrams).transpose()
    testing.labels = np.vstack(testing.labels)
    testing.experiment = np.vstack(testing.experiment)
    testing.n = np.shape(testing.index)

    return [training, testing]


def mad_outlier(data, thresh=3.5):
    """
    Use Median Absolute Deviation method, default in MATLAB
    :param data: array of voltammetry data
    :param thresh: threshold for MAD outlier detection
    :return: exclude_index: indices of outliers
    """
    MAD = robust.mad(data[np.isfinite(data)])
    exclude_index = np.where((data < data.median() - thresh * MAD) | (data > data.median() + 3.5 * MAD))
    return exclude_index


def leave_out_concentration(training, chemIx=0, leave_out_concentration=0):
    idx_lo = np.where(np.array(training.labels[:,0]) != leave_out_concentration)[0]
    training.labels = training.labels[idx_lo][:]
    training.vgrams = training.vgrams[idx_lo][:]
    return training

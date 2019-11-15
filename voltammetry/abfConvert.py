import os
import glob
import pyabf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import h5py
import matplotlib as mpl
import shutil
from numpy.core.multiarray import ndarray

import matplotlib.colors as colors

class Data:
    """
    Experimental data taken from directory of abf files.
    """

    def __init__(self, data_path):
        """
        Define class and class functions
        :param data_path: path containing abf files
        """
        [self.Voltammogram, self.CMD, self.stTime, self.name, self.samplingRate, self.sweep_count,
         self.sweep_point_count] = loadData(data_path)

    def _plotVoltammograms(self, CMAP=cmx.jet, fnY=lambda x: x):
        """
        Plot raw voltammogram data.
        :param CMAP: color map defined in matplotlib.cm (default = jet)
        :param fnY: function applied to voltammograms (defualt = lambda x:x)
        :return: fig: figure object
        """
        fig = plt.figure()
        N = np.shape(self.Voltammogram)[2]  # Number of experiments run
        vgram_colors = CMAP(np.linspace(0, 1, N))
        for i in range(1, N+1):
            vgrams = np.asarray(self.Voltammogram[:, :, i-1])
            Rn = vgrams.shape[1]
            mid = int(np.ceil(Rn / 2))
            y = np.transpose(fnY(vgrams[:, mid]))
            x = 1000 * np.array(range(self.sweep_point_count))/self.samplingRate
            plt.plot(x[0:len(y)], y, color=vgram_colors[i-1])
        plt.title(self.name)
        plt.xlabel('Time (ms)')
        plt.ylabel('current (nA)')
        plt.axis('tight')
        norm = colors.Normalize(vmin=1, vmax=N)
        sm = cmx.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        bounds = np.arange(0.5,N+1.5)
        cbar = plt.colorbar(sm, ticks=np.linspace(0, N+1, N+2), label='experiment #', boundaries=bounds)
        cbar.set_ticklabels(range(1,N+1))
        return fig


# noinspection PyProtectedMember
def abf2H5(abf_path):
    """
    Convert ABF to H5
    :param abf_path: string, abf directory path
    :return: Voltammogram: list, Voltammetry data from directory
    :return: CMD: list, forcing function from abf data
    :return: stTime: array, start time of each file
    :return: abf_name: string with base name of directory
    :return: sampling_rate: integer, number of samples per second
    """
    abf_name = os.path.basename(abf_path)
    if abf_name == '':
        abf_name = os.path.basename(os.path.split(abf_path)[0])
    out_dir = os.path.split(abf_path)[0]

    # Combine data from abf files in given path
    abf_glob = sorted(glob.glob(abf_path + "/*.abf"))  # collection of files in directory
    num_files = len(abf_glob)  # number of abf files in directory
    abf_0 = pyabf.ABF(abf_glob[0])
    sweep_count = abf_0.sweepCount  # Number of sweeps (max 10000)
    sweep_point_count = abf_0.sweepPointCount  # Number of points in sweep (97 Hz = 1032)
    Voltammogram = np.empty((sweep_point_count, sweep_count))
    CMD = np.empty((sweep_point_count, sweep_count))
    for i in range(num_files):
        abf = pyabf.ABF(abf_glob[i])
        abf_file_name = os.path.splitext(os.path.basename(abf.abfFilePath))[0]
        stTime = abf._headerV2.uFileStartTimeMS
        Voltammogram[:, :] = np.asarray(np.reshape(abf.data[0, :], (sweep_point_count, -1), order='F'))
        CMD[:, :] = np.asarray(np.reshape(abf.data[1, :], (sweep_point_count, -1), order='F'))
        with h5py.File(os.path.join(out_dir, abf_file_name + '.h5'), 'w') as f:
            dset_vgram = f.create_dataset("Voltammogram", data=Voltammogram)
            dset_cmd = f.create_dataset("CMD", data=CMD)
            f.attrs["stTimeMS"] = stTime
            f.attrs['samplingRate'] = abf.dataRate
            f.attrs['sweepCount'] = abf.sweepCount
            f.attrs['sweepPointCount'] = abf.sweepPointCount
            f.attrs['expName'] = abf_name


def loadData(h5_path):
    """
    :param h5_path: string, h5 data directory path
    :return: Voltammogram: list, Voltammetry data from directory
    :return: CMD: list, forcing function from abf data
    :return: stTime: array, start time of each file
    :return: abf_name: string with base name of directory
    :return: sampling_rate: integer, number of samples per second
    """
    h5_name = os.path.basename(h5_path)
    # combine data from hdf5 files in given path
    h5_glob = sorted(glob.glob(h5_path + "/*.h5")) # collection of files in directory
    if not h5_glob:
        print('No h5 files found.')
        pass
    else:
        num_files = len(h5_glob) # number of files in directory
        h5_0 = h5py.File(h5_glob[0], 'r')
        sweep_count = h5_0.attrs['sweepCount']
        sweep_point_count = h5_0.attrs['sweepPointCount']
        sampling_rate = h5_0.attrs['samplingRate']
        Voltammogram = np.empty((sweep_point_count, sweep_count, num_files))
        CMD = np.empty((sweep_point_count, sweep_count, num_files))
        stTime = np.empty(num_files)
        for i in range(num_files):
            h5 = h5py.File(h5_glob[i], 'r')
            stTime = h5.attrs['stTimeMS']
            Voltammogram[:, :, i] = h5['Voltammogram']
            CMD[:, :, i] = h5['CMD']
    return[Voltammogram, CMD, stTime, h5_name, sampling_rate, sweep_count, sweep_point_count]

# def save_ABF(abf_path, overwrite=False):
#     """
#     Save abf files to csv directory
#     :param abf_path: string, Path of csv output files Files
#     :param overwrite: boolean, replace files saved in output directory
#     :return:
#     """
#     outDir = ''.join((abf_path, '/OUT'))
#
#     # Options for saving files
#     if not os.path.isdir(outDir):
#
#         print('Saving data to ', outDir)
#         print('...')
#         os.makedirs(outDir)
#         _writeCSV(abf_path, outDir)
#         print('Done.')
#     else:
#         if overwrite:
#
#             print('Removing old files...')
#             shutil.rmtree(outDir)
#             os.makedirs(outDir)
#             print('Saving data to ', outDir)
#             print('...')
#             _writeCSV(abf_path, outDir)
#             print('Done.')
#         else:
#             print('Files already exist -- Not overwriting')
#
# #  TODO: Parse substring for saving model
#  def parse_file_name(abfpath):

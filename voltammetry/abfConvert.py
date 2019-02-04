import os
import glob
import pyabf
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import shutil

class Vgramdata:
    """
    Data collected for calibration
    """

    def __init__(self, abfpath):
        """Load ABF data for calibration"""
        [self.Voltammogram, self.CMD, self.stTime] = loadABF(abfpath)

    def _plotVoltammograms(self, CMAP=cmx.jet, fnY=lambda x: x):
        """Plot voltammograms"""
        N = np.shape(self.Voltammogram)[0]  # Number of experiments run
        for i in range(N, 0, -1):
            vgrams = np.asarray(self.Voltammogram[i - 1])
            Rn = vgrams.shape[1]
            mid = int(np.ceil(Rn / 2))
            y = (fnY(vgrams[mid, :]))
            x = range(len(y))
            plt.plot(x, y, color=CMAP(float(i / N)))
        norm = colors.Normalize(vmin=1, vmax=N)
        sm = cmx.ScalarMappable(cmap=CMAP, norm=norm)

        cbar = plt.colorbar(sm, ticks=np.linspace(1, N, N), label='experiment #')
        plt.title('voltammograms')
        plt.xlabel('sample #')
        plt.ylabel('current (nA)')
        plt.axis('tight')
        plt.show()


def loadABF(abfpath):
    """Convert directory of ABF files to numpy arrays."""
    stTime = []
    CMD = []
    Voltammogram = []
    # Combine data from abf files in given path
    for abfFile in glob.glob(abfpath + "/*.abf"):
        abf = pyabf.ABF(abfFile)
        abfh = abf._headerV2
        numSamples = abf.sweepPointCount
        stTime.append(abfh.uFileStartTimeMS)
        CMD_step = []
        Voltammogram_step = []
        for sweepNumber in range(abf.sweepCount):
            abf.setSweep(sweepNumber)
            CMD_step.append(np.asarray(abf.sweepX))
            Voltammogram_step.append(np.asarray(abf.sweepY))
        CMD.append(CMD_step)
        Voltammogram.append(Voltammogram_step)

    Voltammogram = np.swapaxes(Voltammogram, 0, 2)
    CMD = np.swapaxes(CMD, 0, 2)
    return [Voltammogram, CMD, stTime]


def saveABF(abfpath, overwrite=False):
    """Save files as CSVs."""
    outDir = ''.join((abfpath, '/OUT'))

    # Options for saving files
    if not os.path.isdir(outDir):

        print('Saving data to ', outDir)
        print('...')
        os.makedirs(outDir)
        _writeCSV(abfpath, outDir)
        print('Done.')
    else:
        if overwrite:

            print('Removing old files...')
            shutil.rmtree(outDir)
            os.makedirs(outDir)
            print('Saving data to ', outDir)
            print('...')
            _writeCSV(abfpath, outDir)
            print('Done.')
        else:
            print('Files already exist -- Not overwriting')


def _writeCSV(abfpath, outDir):
    """Write CSV files"""
    stTime_fName = ''.join(('stTime.csv'))
    stTime = []
    for abfFile in glob.glob(abfpath + "/*.abf"):
        fPrefix = os.path.splitext(abfFile)[0][-4:]
        CMD_fName = ''.join(('CMD_', fPrefix, '.csv'))
        Voltammogram_fName = ''.join(('Voltammogram_', fPrefix, '.csv'))
        abf = pyabf.ABF(abfFile)
        abfh = abf._headerV2
        numSamples = abf.sweepPointCount
        CMD = []
        Voltammogram = []
        stTime.append(abfh.uFileStartTimeMS)

        for sweepNumber in range(abf.sweepCount):
            abf.setSweep(sweepNumber)
            CMD.append(np.asarray(abf.sweepX))
            Voltammogram.append(np.asarray(abf.sweepY))

        V = map(list, zip(*Voltammogram))
        C = map(list, zip(*CMD))
        T = np.asarray(stTime)
        # Write forcing functions to csv
        with open(os.path.join(outDir, CMD_fName), "w") as f:
            writer = csv.writer(f)
            writer.writerows(C)
        # Write voltammograms to csv
        with open(os.path.join(outDir, Voltammogram_fName), "w") as f:
            writer = csv.writer(f)
            writer.writerows(V)
            # Write time to csv
            np.savetxt(os.path.join(outDir, stTime_fName), T, delimiter=",")

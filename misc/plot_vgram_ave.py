import voltammetry
import sys
import matplotlib.pyplot as plt
import numpy as np

abfpath = sys.argv[1]
vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabels(abfpath, 'run.csv')
data = voltammetry.PreprocessedData(vg.Voltammogram, labels, trainingSampleSize=125)
testing_vgram_rms = np.sqrt((data.testing.vgrams ** 2).mean(axis=1))
brk = np.where(np.diff(np.sum(data.testing.labels, axis=1)))[0]
testing_vgram_rms[brk] = np.nan

plt.plot(testing_vgram_rms)

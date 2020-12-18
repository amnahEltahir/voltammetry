import voltammetry
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
plt.style.use('ggplot')
plt.rcParams['axes.facecolor']=[1,1,1]
plt.rcParams['axes.edgecolor']='k'

abfpath = sys.argv[1]
vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabels(abfpath, 'run.csv')
idx = np.loadtxt('/Users/amnah/Desktop/wetlab_data/DIssertation_Datasets/misc/10p_sweep_idx.txt',dtype=np.uint16)
vg.Voltammogram = vg.Voltammogram[idx]
vg.CMD = vg.CMD[idx]
data = voltammetry.PreprocessedData(vg.Voltammogram, labels,window_size=425,trainingSampleSize=125,corr_over=True)
# only dopamine


bestAlpha = voltammetry.best_alpha(data.training)

#bestAlpha = 1.0
t = vg.sweep_point_count * data.testing.index / vg.samplingRate
cvFit = voltammetry.train_analyte(data.training, alpha=bestAlpha,parallel=8)#,fnY=lambda x: np.abs(np.fft.rfft(x)))
predictions = voltammetry.test_analyte(data.testing, cvFit)#,fnY=lambda x: np.abs(np.fft.rfft(x)))
with h5py.File(abfpath.split()[0]+'10percent_DA_results.h5','w') as f:
    f.create_group('raw')
    f.create_dataset('raw/labels',data=labels.labels)
    f.create_dataset('raw/vgrams',data=vg.Voltammogram)
    f.create_dataset('raw/CMD',data=vg.CMD)
    f.create_dataset('raw/idx',data=idx)
    f.attrs['targetAnalyte'] = np.string_(labels.targetAnalyte)
    f.create_group('results')
    f.create_dataset('results/predictions',data=predictions)
    f.create_dataset('results/actual',data=data.testing.labels)
# for chemIx in range(len(labels.targetAnalyte)):
#      stats = voltammetry.calcStepStats(chemIx, predictions, data.testing.labels)
#      calFig = voltammetry.plot_Calibration(t, predictions, data.testing.labels, labels.targetAnalyte, chemIx, stats)
#      calFig.suptitle(vg.name)
#      plt.show()

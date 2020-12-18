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
data = voltammetry.PreprocessedData(vg.Voltammogram[16:1016], labels,window_size=425,trainingSampleSize=125,corr_over=True)

# idx = np.where(data.training.labels[:,1] ==0)
# data.training.labels = data.training.labels[idx]
# data.training.vgrams = data.training.vgrams[idx]
# idx_test = np.where(data.testing.labels[:,1]==0)
# data.testing.labels = data.testing.labels[idx_test]
# data.testing.vgrams = data.testing.vgrams[idx_test]

#bestAlpha = voltammetry.best_alpha(data.training)

bestAlpha = 1.0
t = vg.sweep_point_count * data.testing.index / vg.samplingRate

cvFit = voltammetry.train_analyte(data.training, alpha=bestAlpha,parallel=8)
predictions = voltammetry.test_analyte(data.testing, cvFit)
with h5py.File(abfpath.split()[0]+'_DA_pH_results.h5','w') as f:
    f.create_group('training')
    f.create_dataset('training/vgrams', data=data.training.vgrams)
    f.create_dataset('training/labels', data=data.training.labels)
    f.create_group('testing')
    f.create_dataset('testing/vgrams', data=data.testing.vgrams)
    f.create_dataset('testing/labels', data=data.testing.labels)
    f.attrs['targetAnalyte'] = np.string_(labels.targetAnalyte)
    f.create_group('results')
    f.create_dataset('results/predictions', data=predictions)
# for chemIx in range(len(labels.targetAnalyte)):
#      stats = voltammetry.calcStepStats(chemIx, predictions, data.testing.labels)
#      calFig = voltammetry.plot_Calibration(t, predictions, data.testing.labels, labels.targetAnalyte, chemIx, stats)
#      calFig.suptitle(vg.name)
#      plt.show()

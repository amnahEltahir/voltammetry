import voltammetry
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py


abfpath = sys.argv[1]
vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabels(abfpath, 'run.csv')
idx =np.where((labels.labels['5HT']==0) | (labels.labels['DA'] == 0))
labels.labels = labels.labels.iloc[idx]
data = voltammetry.PreprocessedData(np.squeeze(vg.Voltammogram[:,:,idx]), labels,trainingSampleSize=125)
bestAlpha = 1
t = vg.sweep_point_count * data.testing.index / vg.samplingRate
cvFit = voltammetry.train_analyte(data.training, alpha=bestAlpha)
predictions = voltammetry.test_analyte(data.testing, cvFit)
"""
with h5py.File(vg.name+"_edge.h5",'w') as f:
    data1 = f.create_dataset('predictions', data=predictions)
    data2 = f.create_dataset('labels', data=data.testing.labels)


for chemIx in range(len(labels.targetAnalyte)):
    stats = voltammetry.calcStepStats(chemIx, predictions, data.testing.labels)
    calFig = voltammetry.plot_Calibration(t, predictions, data.testing.labels, labels.targetAnalyte, chemIx, stats)
    calFig.suptitle(vg.name)
    plt.show()
"""
stats = voltammetry.calcStepStats(0, predictions, data.testing.labels)
calFig = voltammetry.plot_Calibration(t, predictions, data.testing.labels, labels.targetAnalyte, 0, stats)
calFig.suptitle(vg.name)
plt.show()
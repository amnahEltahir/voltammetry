import voltammetry
import sys
import matplotlib.pyplot as plt
import numpy as np
import random as rand

plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = [1, 1, 1]
plt.rcParams['axes.edgecolor'] = 'k'
random_seed = 0
rand.seed(random_seed)

abfpath = sys.argv[1]
vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabels(abfpath, 'run.csv')
data = voltammetry.PreprocessedData(vg.Voltammogram, labels, trainingSampleSize=125)
# find unique tuples
unique_tuples = np.unique(data.training.labels, axis=0)
numC = len(unique_tuples)
# set up folds
nfolds = int(7)
folds = np.zeros((numC))
cPerF = int(numC / nfolds)
folds[0:77] = np.random.permutation(np.repeat(np.arange(nfolds), cPerF))
foldid = np.zeros((len(data.training.labels), 1), dtype=np.int8)
for i, tup in enumerate(unique_tuples):
    idx = np.where((data.training.labels[:,0] == tup[0]) & (data.training.labels[:,1] == tup[1]) & (data.training.labels[:,2] == tup[2]))
    foldid[idx] = folds[int(i)]
# find best alpha
bestAlpha = 1.0
# bestAlpha = voltammetry.best_alpha(data.training)
t = vg.sweep_point_count * data.testing.index / vg.samplingRate
cvFit = voltammetry.train_analyte(data.training, alpha=bestAlpha, nfolds=nfolds, foldid=foldid, parallel=8)
predictions = voltammetry.test_analyte(data.testing, cvFit)
for chemIx in range(len(labels.targetAnalyte)):
    stats = voltammetry.calcStepStats(chemIx, predictions, data.testing.labels)
    calFig = voltammetry.plot_Calibration(t, predictions, data.testing.labels, labels.targetAnalyte, chemIx, stats)
    calFig.suptitle(vg.name)
    plt.show()

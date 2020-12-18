import voltammetry
import sys
import h5py
import os
import glmnet_python
from glmnet_python import cvglmnetCoef
import numpy as np


abfpath = sys.argv[1]
abf_name = os.path.basename(abfpath)

out_dir = os.path.split(abfpath)[0]

vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabels(abfpath, 'run.csv')
#idx =np.where((labels.labels['NE']==0) | (labels.labels['DA'] == 0))
#labels.labels = labels.labels.iloc[idx]
#data = voltammetry.PreprocessedData(np.squeeze(vg.Voltammogram[:,:,idx]), labels,trainingSampleSize=125)
data = voltammetry.PreprocessedData(vg.Voltammogram, labels,trainingSampleSize=125)
bestAlpha = 1.0
t = vg.sweep_point_count * data.testing.index / vg.samplingRate
cvFit = voltammetry.train_analyte(data.training, alpha=bestAlpha)
predictions = voltammetry.test_analyte(data.testing, cvFit)
SNR = np.empty(0)
RMSE = np.empty(0)
fullSNR = np.empty(0)
fullRMSE = np.empty(0)

for chemIx in range(len(labels.targetAnalyte)):
    stats = voltammetry.calcStepStats(chemIx, predictions, data.testing.labels)
    coefs = cvglmnetCoef(cvFit, 'lambda_min')
    SNR = np.concatenate((SNR,stats.prediction_SNR))
    RMSE = np.concatenate((RMSE, stats.prediction_RMSE))
    fullSNR = np.hstack((fullSNR,stats.fullSNR))
    fullRMSE = np.hstack((fullRMSE, stats.fullRMSE))
#with h5py.File(abf_name + "_a_" + str(int(100*bestAlpha)) + '_' + '_edge.h5', 'w') as f:
with h5py.File(abf_name + "_a_" + str(int(100 * bestAlpha)) + '.h5', 'w') as f:

    f.create_dataset("trainingV", data=data.training.vgrams)
    f.create_dataset("testingV", data=data.testing.vgrams)
    f.attrs["coefs"] = coefs
    f.attrs['targetAnalyte'] = np.string_(labels.targetAnalyte)
    f.create_dataset("predictions", data=predictions)
    f.create_dataset("actual", data=data.testing.labels)
    f.create_dataset("trainingLabels", data=data.training.labels)
    f.attrs['SNR'] = np.array(SNR)
    f.attrs['RMSE'] = np.array(RMSE)
    f.attrs['full_SNR'] = np.array(fullSNR)
    f.attrs['full_RMSE'] = np.array(fullRMSE)
    f.attrs['CMD'] = vg.CMD[:, 0, 0]



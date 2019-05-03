import voltammetry
import sys
import matplotlib.pyplot as plt



abfpath = sys.argv[1]
loc = sys.argv[2]
vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabels(abfpath, 'run.csv')
data = voltammetry.PreprocessedData(vg.Voltammogram, labels, trainingSampleSize=125, window_size=1500)
data.training = voltammetry.leave_out_concentration(data.training, leave_out_concentration=loc)
bestAlpha = voltammetry.best_alpha(data.training)
t = vg.sweep_point_count * data.testing.index / vg.samplingRate
cvFit = voltammetry.train_analyte(data.training, alpha=bestAlpha)
predictions = voltammetry.test_analyte(data.testing, cvFit)
for chemIx in range(len(labels.targetAnalyte)):
    stats = voltammetry.calcStepStats(chemIx, predictions, data.testing.labels)
    calFig = voltammetry.plot_Calibration(t, predictions, data.testing.labels, labels.targetAnalyte, chemIx, stats)
    calFig.suptitle(vg.name)
    plt.show()
import voltammetry
import sys
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

abfpath = sys.argv[1]
vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabels(abfpath, 'run.csv')
data = voltammetry.PreprocessedData(vg.Voltammogram, labels)
bestAlpha = voltammetry.best_alpha(data.training)
t = np.linspace(0, 1032 * len(data.testing.labels) / vg.samplingRate, len(data.testing.labels))
cvFit = voltammetry.train_analyte(data.training, alpha=bestAlpha)
predictions = voltammetry.test_analyte(data.testing, cvFit)
for chemIx in range(len(labels.targetAnalyte)):
    stats = voltammetry.calcStepStats(chemIx, predictions, data.testing.labels)
    calFig = voltammetry.plot_Calibration(t, predictions, data.testing.labels, labels.targetAnalyte, chemIx, stats)
    calFig.suptitle(vg.name)
    plt.show()

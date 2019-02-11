import voltammetry
import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

abfpath = sys.argv[1]
vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabels(abfpath, 'run.csv')
data = voltammetry.PreprocessedData(vg.Voltammogram, labels)
bestAlpha = voltammetry.best_alpha(data.training)
cvFit = voltammetry.train_analyte(data.training, alpha=bestAlpha)
predictions = voltammetry.test_analyte(data.testing, cvFit)
stats = voltammetry.calcStepStats(0, predictions, data.testing.labels)

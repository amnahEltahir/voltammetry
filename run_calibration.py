import voltammetry
import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

abfpath = sys.argv[1]
vg = voltammetry.Data(abfpath)
labels = voltammetry.Mulabrm els(abfpath, 'run.csv')
data = voltammetry.PreprocessedData(vg.Voltammogram, labels)
cvFit = voltammetry.train_analyte(data.training, alpha=0)
predictions = voltammetry.test_analyte(data.testing, cvFit)
stats = voltammetry.calcStepStats('DA', predictions, data.testing.labels)

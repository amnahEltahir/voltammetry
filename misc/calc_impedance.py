import os
import voltammetry
import sys
import numpy as np


abfpath = sys.argv[1]
abfname = os.path.basename(os.path.split(abfpath)[0])
if os.path.isdir(abfpath):
    pass
else:
    print('Not a valid path!!!')
    exit(1)

vg = voltammetry.Data(abfpath)
len_measure = 1500
sweep_start = int(np.ceil(vg.sweep_point_count / 2))
sweep_stop = sweep_start + len_measure
V = vg.CMD[sweep_start:sweep_stop, 2, 0:5]
I = vg.Voltammogram[sweep_start:sweep_stop, 2, 0:5]
impedance = np.zeros(5)
for j in range(5):
    V_rms = np.sqrt(np.mean(V[:, j] ** 2))
    I_rms = np.sqrt(np.mean(I[:, j] ** 2))
    impedance[j] = V_rms/I_rms


print(abfname, ' '.join(map("{:.3f}".format, impedance)))
exit(0)

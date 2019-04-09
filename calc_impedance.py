import os
import voltammetry
import sys
import numpy as np


abfpath = sys.argv[1]

if os.path.isdir(abfpath):
    print(abfpath)
else:
    print('Not a valid path!!!')
    exit(1)

vg = voltammetry.Data(abfpath)
len_measure = 1500
sweep_start = int(np.ceil(vg.sweep_point_count / 2))
sweep_stop = sweep_start + len_measure
v = vg.CMD[sweep_start:sweep_stop, 2, 0]
i = vg.Voltammogram[sweep_start:sweep_stop, 2, 0]
V_rms = np.sqrt(np.mean(v ** 2))
I_rms = np.sqrt(np.mean(i ** 2))
impedance = V_rms/I_rms
print("Impedance = ", '{:.3f}'.format(impedance), "milli-Ohm")
exit(0)

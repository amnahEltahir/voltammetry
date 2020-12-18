import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.gridspec import GridSpec
import glob
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 24})

concentrations = np.arange(0,2750,50)
results_glob = sorted(glob.glob("2019*.h5"))
num_datafiles=len(results_glob)
SNR=[]
RMSE=[]


for i in range(3):

    for j in range(len(results_glob)):
        data = h5py.File(results_glob[j])
        SNR.append(data.attrs['full_SNR'][i])
        RMSE.append(data.attrs['full_RMSE'][i])
plt.figure()

plt.scatter(np.arange(0,20),SNR[0:20],s=50,color='#1f77b4')
plt.scatter(np.arange(0,20),SNR[20:40],s=50,color='#b49e1f')
plt.scatter(np.arange(0,20),SNR[40:60],s=50,color='#3ebd30')
plt.title("Prediction SNR")
plt.xticks(np.arange(0,20),('A_1','A_2','B_1','B_2','C_1','C_2','D_1','D_2','E_1','E_2','F_1','F_2','G_1',
                                   'G_2','H_1','H_2','I_1','I_2','J_1','J_2'),rotation=45)
plt.legend(('DA','5HT','pH'))
plt.ylabel('SNR (dB)')

plt.figure()

plt.scatter(np.arange(0,20),RMSE[0:20],s=50,color='#1f77b4')
plt.scatter(np.arange(0,20),RMSE[20:40],s=50,color='#b49e1f')
plt.scatter(np.arange(0,20),RMSE[40:60],s=50,color='#3ebd30')
plt.title("Prediction RMSE")
plt.xticks(np.arange(0,20),('A_1','A_2','B_1','B_2','C_1','C_2','D_1','D_2','E_1','E_2','F_1','F_2','G_1',
                                   'G_2','H_1','H_2','I_1','I_2','J_1','J_2'),rotation=45)
plt.legend(('DA','5HT','pH'))
plt.ylabel('RMSE (nM)')
plt.axis('tight')
plt.show()




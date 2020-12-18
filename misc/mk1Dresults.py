import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.gridspec import GridSpec
plt.style.use('ggplot')

data_file = sys.argv[1] 
data = h5py.File(data_file)
num_samples = data['predictions'].shape[0]
X = np.arange(num_samples)

Y = data['predictions']
L = np.array(data['actual'])
targetAnalyte = data.attrs['targetAnalyte']

f_SNR = data.attrs['SNR']
f_RMSE = data.attrs['RMSE']
f_fullSNR = data.attrs['full_SNR']
f_fullRMSE = data.attrs['full_RMSE']
chemLabel = []
chem_indeces = []
SNR = []
RMSE = []
fullSNR = []
fullRMSE = []
ph_index = []
for i in range(len(targetAnalyte)):
    if targetAnalyte[i].decode('UTF-8') != 'pH':
        chemLabel.append(targetAnalyte[i].decode('UTF-8'))
        chem_indeces.append(i)
    else:
        ph_index = i
    nSteps = len(np.unique(L[:,i]))
    SNR.append(f_SNR[np.arange(nSteps)])
    RMSE.append(f_RMSE[np.arange(nSteps)])
    fullSNR.append(f_fullSNR[i])
    fullRMSE.append(f_fullRMSE[i])
    f_SNR = np.delete(f_SNR,np.arange(nSteps))
    f_RMSE = np.delete(f_RMSE,np.arange(nSteps))

num_analytes = len(chemLabel)
Figure = plt.figure()

# Plot Predictions
def analyte_specs(chemL):
    units = ''
    labColor ='y'
    if chemL == 'NE':
        chemL = 'NE'
        units = '(nM)'
        labColor = '#b4531f'
    if (chemL == 'Dopamine') | (chemL == 'DA'):
        chemL = 'DA'
        units = '(nM)'
        labColor = '#1f77b4'
    if (chemL == 'Serotonin') | (chemL == '5HT'):
        chemL = '5HT'
        units = '(nM)'
        labColor = '#b49e1f'
    if chemL == '5HIAA':
        chemL = '5HIAA'
        units = '(nM)'
        labColor = '#871fb4'
    if chemL == 'pH':
        chemL = 'pH'
        units = ''
        labColor = '#3ebd30'
    return(chemL,units,labColor)

## Plot results
if ph_index:
    gs = GridSpec(9, 5)
    ax_ph = plt.subplot(gs[8:9, :])
    chemL = 'pH'
    [chemID,units,labColor] = analyte_specs(chemL)
    muLabel = ''.join([chemID, units])
    plt.scatter(X, Y[:, ph_index], marker='.', color=labColor,alpha=0.5)
    L[np.where(np.diff(L[:, ph_index]))] = np.nan
    plt.plot(X, L[:, ph_index], color='k', linewidth=1.0)
    plt.xlabel('Sweep #')
    plt.ylabel(''.join(['pH']))
    plt.axis('tight')
    plt.title('pH Predictions')
else:
    gs = GridSpec(7,5)

# Plot Predictions
ax1 = plt.subplot(gs[1:4, :])
for chemIx in chem_indeces:
    chemL = chemLabel[chemIx]
    [chemID,units,labColor] = analyte_specs(chemL)
    muLabel = ''.join([chemID, units])
    plt.scatter(X, Y[:, chemIx], marker='.', color=labColor,alpha=0.4)
    L[np.where(np.diff(L[:, chemIx]))] = np.nan
    plt.plot(X, L[:, chemIx], color='k', linewidth=3.0)
    plt.fill_between(X,np.squeeze(Y[:,chemIx]),facecolor=labColor,alpha=0.2)
    plt.xlabel('Sweep #')
    plt.ylabel(''.join(['Concentration (nM)']))
    plt.axis('tight')
    plt.title('Neurotransmitter Predictions')
## RMSE subplot

ax2 = plt.subplot(gs[5:7, 0:2])
for chemIx in chem_indeces:
    chemL = chemLabel[chemIx]
    [chemID,units,labColor] = analyte_specs(chemL)
    y = RMSE[chemIx]
    x = np.unique(L[:,chemIx])
    xx = x[~np.isnan(x)]
    ax2.scatter(xx, y, color=labColor)
    ax2.plot(plt.xlim(), [fullRMSE[chemIx], fullRMSE[chemIx]], linestyle='--', markersize=1,
        color=labColor)
    plt.title('RMSE')
    plt.ylabel(''.join(['RMSE (nM)']))
    plt.xlabel(''.join(['concentration (nM)']))
    plt.axis('tight')
# Plot SNR
ax3 = plt.subplot(gs[5:7, 3:5])
for chemIx in chem_indeces:
    chemL = chemLabel[chemIx]
    [chemID,units,labColor] = analyte_specs(chemL)
    y = SNR[chemIx]
    x = np.unique(L[:,chemIx])
    xx = x[~np.isnan(x)]
    ax3.scatter(xx, y, color=labColor)
    ax3.plot(plt.xlim(), [fullSNR[chemIx], fullSNR[chemIx]], linestyle='--', markersize=1,
        color=labColor)
    plt.title('SNR')
    plt.xlabel(muLabel)
    plt.ylabel('SNR (dB)')
    plt.axis('tight')
    plt.xlabel('concentration (nM)')
    plt.ylabel('SNR (dB)')
    plt.axis('tight')
plt.show()



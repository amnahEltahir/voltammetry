import os
import pyabf
import glob
import sys
import h5py
import numpy as np


abf_path = sys.argv[1] # path of directory containing abf data files
replace_str = sys.argv[2] # path of substring to be replaced in file name
ch1 = sys.argv[3] # first channel name
ch2 = sys.argv[4] # second channel name


out_path = os.path.split(abf_path)[0] # name of output path
abf_name = os.path.basename(abf_path) # name of experiment file name

channels = [ch1, ch2] # make list of channel names


abf_glob = sorted(glob.glob(abf_path + "/*.abf"))  # collection of files in directory
num_files = len(abf_glob)  # number of abf files in directory
abf_0 = pyabf.ABF(abf_glob[0])
sweep_count = abf_0.sweepCount  # Number of sweeps (max 10000)
sweep_point_count = abf_0.sweepPointCount  # Number of points in sweep (97 Hz = 1032)
sampling_rate = abf_0.dataRate
channel_count = abf_0.channelCount

# Check to make sure that directory name has substring to be replaced
if replace_str not in abf_name:
    print('Unable to rename files. Check if <' + replace_str + '>is in the directory name.')
    exit(3)
# Check if the data only has one channel
if channel_count == 2:
    print('Only one channel in file. Nothing to do.')
    exit(2)
# Split channels
elif channel_count == 4:
    print('Splitting channels.')
    for i in range(2):
        # Make output directories, making sure they don't already exist
        ch_name = abf_name.replace(replace_str, channels[i])
        ch_dir = os.path.join(out_path, ch_name)
        if not os.path.isdir(ch_dir):
            os.makedirs(ch_dir)
        else:
            if os.listdir(ch_dir):
                print(ch_dir + ' Directory not empty. Channels already split.')
                exit(1)
        stTime = np.empty(num_files)

    for j in range(num_files):
        abf = pyabf.ABF(abf_glob[j])

        for i in range(2):
            ch_name = abf_name.replace(replace_str, channels[i])
            ch_dir = os.path.join(out_path, ch_name)
            Voltammogram = np.empty((sweep_point_count, sweep_count))
            CMD = np.empty((sweep_point_count, sweep_count))
            abf_file_name = os.path.splitext(os.path.basename(abf.abfFilePath).replace(replace_str,channels[i]))[0]
            stTime = abf._headerV2.uFileStartTimeMS
            Voltammogram[:, :] = np.asarray(np.reshape(abf.data[i*2, :], (sweep_point_count, -1), order='F'))
            CMD[:, :] = np.asarray(np.reshape(abf.data[i*2+1, :], (sweep_point_count, -1), order='F'))
            print(os.path.join(ch_dir, abf_file_name + '.h5'))
            with h5py.File(os.path.join(ch_dir, abf_file_name + '.h5'), 'w') as f:
                dset_vgram = f.create_dataset("Voltammogram", data=Voltammogram)
                dset_cmd = f.create_dataset("CMD", data=CMD)
                f.attrs["stTimeMS"] = stTime
                f.attrs['samplingRate'] = abf.dataRate
                f.attrs['sweepCount'] = abf.sweepCount
                f.attrs['sweepPointCount'] = abf.sweepPointCount
                f.attrs['expName'] = abf_name

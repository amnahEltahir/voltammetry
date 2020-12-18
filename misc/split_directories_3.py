import os
import sys
import glob
from shutil import copyfile


# Check that this is the right type of directory to split
path = sys.argv[1]
if ("YYY" not in path) | (not os.path.isdir(path)):
    print("Not a valid directory!!!")
    exit(1)
# Get relevant information about the path for making directories
data_parent = os.path.dirname(path)
data_base = os.path.basename(path)
[prefix, suffix] = data_base.split('YYY', 1)
r00 = prefix + 'FSCV_10Hz_100k' + suffix
r01 = prefix + 'uncorrelated_97Hz_100k' + suffix
r02 = prefix + 'uncorrelated_97Hz_100k_25' + suffix
r00_out = data_parent + '/' + r00
r01_out = data_parent + '/' + r01
r02_out = data_parent + '/' + r02

if os.path.isdir(r00_out) | os.path.isdir(r01_out):
    print('This directory has already been split. \n')
    print('Nothing to do.\n')
    exit(2)
else:
    os.makedirs(r00_out)
    os.makedirs(r01_out)
    os.makedirs(r02_out)

# Rename files and copy to appropriate directories
file_list = sorted(glob.glob(path + '/*.abf'))

for file in file_list:
    name = os.path.splitext(os.path.basename(file))[0]
    seq_num = int(name[-4::])
    if seq_num % 3 == 0:
        out_path = r00_out
        new_num = int(seq_num / 3)
        new_str = r00 + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, r00_out + '/' + new_str)
        print(name + '\t' + 'r00' + '\n')
        continue
    if (seq_num - 1) % 3 == 0:
        out_path = r01_out
        new_num = int((seq_num-1) / 3)
        new_str = r01 + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, r01_out + '/' + new_str)
        print(name + '\t' + 'r01' + '\n')

        continue
    else:
        out_path = r02_out
        new_num = int(((seq_num+1)/3) - 1)
        new_str = r02 + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, r02_out + '/' + new_str)
        print(name + '\t' + 'r02' + '\n')


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
uncorrelated_name = prefix + 'uncorrelated_100k_25' + suffix
uncorrelated_name_50 = prefix + 'uncorrelated_100k_10' + suffix
fastRB_name = prefix + 'uncorrelated_1kHz_500k' + suffix
chem001_name = prefix + 'two_pulse_1kHz_500k' + suffix
uncorrelated_out = data_parent + '/' + uncorrelated_name
uncorrelated_50_out = data_parent + '/' + uncorrelated_name_50
fastRB_out = data_parent + '/' + fastRB_name
chem001_out = data_parent + '/' + chem001_name

if os.path.isdir(uncorrelated_out) | os.path.isdir(uncorrelated_50_out):
    print('This directory has already been split. \n')
    print('Nothing to do.\n')
    exit(2)
else:
    os.makedirs(uncorrelated_out)
    os.makedirs(uncorrelated_50_out)
    os.makedirs(fastRB_out)
    os.makedirs(chem001_out)

# Rename files and copy to appropriate directories
file_list = sorted(glob.glob(path + '/*.abf'))

for file in file_list:
    name = os.path.splitext(os.path.basename(file))[0]
    seq_num = int(name[-4::])
    if seq_num % 4 == 0:
        out_path = uncorrelated_out
        new_num = int(seq_num / 4)
        new_str = uncorrelated_name + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, uncorrelated_out + '/' + new_str)
        continue
    if seq_num % 2 == 0:
        out_path = fastRB_out
        new_num = int(seq_num/4)
        new_str = fastRB_name + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, fastRB_out + '/' + new_str)
        continue
    if (seq_num - 1) % 4 == 0:
        out_path = uncorrelated_50_out
        new_num = int((seq_num-1) / 4)
        new_str = uncorrelated_name_50 + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, uncorrelated_50_out + '/' + new_str)
        continue
    else:
        out_path = chem001_out
        new_num = int(((seq_num+1)/4) - 1)
        new_str = chem001_name + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, chem001_out + '/' + new_str)

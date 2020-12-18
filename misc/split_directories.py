import os
import sys
import glob
from shutil import copyfile


# Check that this is the right type of directory to split
path = sys.argv[1]
replace_str = sys.argv[2] # path of substring to be replaced in file name
f1 = sys.argv[3] # first channel name
f2 = sys.argv[4] # second channel name
if (replace_str not in path) | (not os.path.isdir(path)):
    print("Not a valid directory!!!")
    exit(1)
# Get relevant information about the path for making directories
data_parent = os.path.dirname(path)
data_base = os.path.basename(path)
[prefix, suffix] = data_base.split(replace_str, 1)
triangle_name = prefix + f1 + suffix
uncorrelated_name = prefix + f2 + suffix
triangle_out = data_parent + '/' + triangle_name
uncorrelated_out = data_parent + '/' + uncorrelated_name

if os.path.isdir(triangle_out) | os.path.isdir(uncorrelated_out):
    print('This directory has already been split. \n')
    print('Nothing to do.\n')
    exit(2)
else:
    os.makedirs(triangle_out)
    os.makedirs(uncorrelated_out)

# Rename files and copy to appropriate directories
file_list = sorted(glob.glob(path + '/*.abf'))

for file in file_list:
    name = os.path.splitext(os.path.basename(file))[0]
    seq_num = int(name[-4::])
    if seq_num % 2 == 0:
        out_path = triangle_out
        new_num = int(seq_num/2)
        new_str = triangle_name + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, triangle_out + '/' + new_str)
    else:
        out_path = uncorrelated_out
        new_num = int((seq_num-1)/2)
        new_str = uncorrelated_name + "_{0:0>4}".format(new_num) + '.abf'
        copyfile(file, uncorrelated_out + '/' + new_str)

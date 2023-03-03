"""
Collect all the usable file names in a list. Usable meeting the criteria:

1. sampling frequency is a multiple of 256.
"""

import glob
import mne
import argparse
from path import Path

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--set", type=str, help="the set of interest e.g. train/dev/eval")
argParser.add_argument("-l", "--folder_location", choices=['external','local'],type=str,help="use local or external storage")
args = argParser.parse_args()

if args.folder_location == "local":
    path = Path.machine_path + args.set 
else:
    path = Path.ext_path + args.set


counter = 0
file_list = []
list_file = open('file_list.txt', 'w')

for f in glob.glob(path + '/**/*.edf', recursive=True):
    data = mne.io.read_raw_edf(f, infer_types=True)
    if data.info['sfreq'] % 256 == 0:
        file_list.append(f)
        list_file.write(f+"\n")
        counter += 1


list_file.close()

print(counter)
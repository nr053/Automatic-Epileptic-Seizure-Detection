"""
Collect all the usable file names in a list. Usable meeting the criteria:

1. sampling frequency is a multiple of 256.
"""

import glob
import mne
from tqdm import tqdm

counter = 0
file_list = []
list_file = open('file_list_1000.txt' , 'w')

for f in tqdm(glob.glob('/home/migo/TUHP/TUSZ_V2/edf/train' + '/**/*.edf', recursive=True)):
    data = mne.io.read_raw_edf(f, infer_types=True)
    if data.info['sfreq']  == 1000:
        file_list.append(f)
        list_file.write(f+"\n")
        counter += 1


list_file.close()

print(counter)
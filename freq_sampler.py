"""This script iterates through a data set (taken as argument) and prints the different sampling
frequencies along with the corresponding occurence count"""

import mne
import numpy as np
import os
import argparse
import glob
from matplotlib import pyplot as plt


argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--set", type=str, help="specify the data set of interest")
args = argParser.parse_args()

path = os.getcwd() + '/TUSZ_V2/edf/' + args.set

s_freqs = []

for f in glob.glob(path + '/**/*.edf', recursive=True):
        data = mne.io.read_raw_edf(f)
        sampling_freq = data.info['sfreq']
        s_freqs.append(sampling_freq)

set = set(s_freqs)
counts = []

for freq in set:
      counts.append([freq, s_freqs.count(freq)])  


print(counts)
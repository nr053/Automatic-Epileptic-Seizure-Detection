"""
This script iterates through a data set (taken as argument) and prints:
1. the different sampling frequencies along with the corresponding occurence count
2. the low and high pass filters 
Script may be extended to create a list of usable files i.e. files that share the same
sampling rate or a multiple of.
"""

import mne
import numpy as np
import os
import argparse
import glob


argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--set", type=str, help="specify the data set of interest")
args = argParser.parse_args()

path = '/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/' + args.set

s_freqs = []
lowpass_list = []
highpass_list = []

for f in glob.glob(path + '/**/*.edf', recursive=True):
      data = mne.io.read_raw_edf(f)
        
      sampling_freq = data.info['sfreq']
      lowpass = data.info['lowpass']
      highpass = data.info['highpass']
        
      s_freqs.append(sampling_freq)
      lowpass_list.append(lowpass)
      highpass_list.append(highpass)

freq_set = set(s_freqs)
lowpass_set = set(lowpass_list)
highpass_set = set(highpass_list)

sfreq_counts = []
lowpass_counts = []
highpass_counts = []

for freq in freq_set:
      sfreq_counts.append([freq, s_freqs.count(freq)])  

for filter in lowpass_set:
      lowpass_counts.append([filter, lowpass_list.count(filter)])

for filter in highpass_set:
      highpass_counts.append([filter, highpass_list.count(filter)])

print("Sampling frequencies: ", sfreq_counts)
print("Low pass filters : ", lowpass_counts)
print("High pass filters: ", highpass_counts)
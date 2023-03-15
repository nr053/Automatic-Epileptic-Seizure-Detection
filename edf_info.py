"""
This script iterates through a data set (taken as argument) and prints:
1. the different sampling frequencies along with the corresponding occurrence count
2. the low and high pass filters 
3. the electrode positions used 
Script may be extended to create a list of usable files i.e. files that share the same
sampling rate or a multiple of.
"""

import mne
import numpy as np
import os
import argparse
import glob
from collections import Counter
from path import Path

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--set", type=str,
                        default="",
                        help="specify the data set of interest")
argParser.add_argument("-p", "--path", type=str, 
                       choices=["local", "ext"],
                       help="the path location of the parent directory (machine/ext hard drive)")
args = argParser.parse_args()

if args.path == "local":
      path = Path.machine_path + args.set
else:
      path = Path.ext_path + args.set

s_freqs = []      #empty list of sample frequencies
lowpass_list = []       #empty list of lowpass filters
highpass_list = []      #empty list of highpass filters
ch_names_set = set()       #empty set of channel names

cnt = Counter()

for f in glob.glob(path + '/**/*.edf', recursive=True):     #iterate through every edf file
      data = mne.io.read_raw_edf(f, infer_types=True)
        
      sampling_freq = data.info['sfreq']        #extract sampling frequency
      lowpass = data.info['lowpass']            #extract low pass filter
      highpass = data.info['highpass']          #extract high pass filter
        
      s_freqs.append(sampling_freq)       #append sampling frequency to sampling freq list
      lowpass_list.append(lowpass)        #append low pass filter to list
      highpass_list.append(highpass)      #append high pass filter to list

      for chName in data.info['ch_names']:
            chName = chName.removesuffix("-REF")
            chName = chName.removesuffix("-LE")
            ch_names_set.add(chName)
            cnt[chName] += 1



freq_set = set(s_freqs)       #make set from list to remove duplicates
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
print(cnt)
print(len(cnt))
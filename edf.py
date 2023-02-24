"""
This script loads the first .edf file in the training set for inspection.
"""
import mne
import numpy as np



path = '/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaaaac/s001_2002_12_23/02_tcp_le/aaaaaaac_s001_t000.edf' 

data = mne.io.read_raw_edf(path)

sampling_freq = data.info['sfreq']


print(data.info)
"""
Show the distribution of high and low pass filters across all recordings. 
"""

import glob
import mne
from tqdm import tqdm
from collections import Counter
import p_tools
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

file_name = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaapwy/s007_2014_06_17/01_tcp_ar/aaaaapwy_s007_t002.edf"


raw_signal_object = mne.io.read_raw_edf(file_name, infer_types=True, preload=True)
raw_signal_object.filter(1,70) #filter raw object
raw_bipolar = p_tools.apply_montage(raw_signal_object) # raw object, with montage
raw_bipolar_signal = raw_bipolar.get_data()


raw_signal_object2 = mne.io.read_raw_edf(file_name, infer_types=True, preload=True)
raw_bipolar2 = p_tools.apply_montage(raw_signal_object2) # raw object, with montage
raw_bipolar_signal2 = raw_bipolar2.get_data()


x = np.linspace(0,1,1280)

#plt.figure(1)
#filter the raw object before applying the montage
#plt.subplot(411)
plt.plot(x, raw_bipolar_signal[0][:1280], x, raw_bipolar_signal2[0][:1280], 'r-')
plt.legend(['raw', 'filtered'])
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude [microvolts]")
plt.title('mne.raw.io.filter', loc='left')
##filter directly on data before applying the montage
# plt.subplot(412)
# plt.plot(x, raw_bipolar_signal[0][:1280], x, signal_filtered[0][:1280], 'r-')
# plt.legend(['raw', 'filtered'])
# plt.xlabel("Time [seconds]")
# plt.ylabel("Amplitude [microvolts]")
# plt.title('mne.filter.filter_data', loc='left')
# #filter the raw object after applying the montage
# plt.subplot(413)
# plt.plot(x, signal1[0][:1280], x, signal_resampled[0][:1280], 'r-')
# plt.legend(['raw', 'resampled'])
# plt.xlabel("Time [seconds]")
# plt.ylabel("Amplitude [microvolts]")
# plt.title('signal.resample_poly', loc='left')
# #filter directly on data object after applying the montage
# plt.subplot(414)
# plt.plot(x, signal1[0][:1280], x, signal_filtered_resampled[0][:1280], 'r-')
# plt.legend(['raw', 'filtered + resampled'])
# plt.xlabel("Time [seconds]")
# plt.ylabel("Amplitude [microvolts]")
# plt.title('mne.filter.filter_data + signal.resample-poly', loc='left')
plt.show()

def load_data(file_name, epoch_length):
    raw_signal_object = mne.io.read_raw_edf(file_name, infer_types=True, preload=True)
    raw_signal_object.filter(1,70)
    #sfreq = raw_signal_object.info['sfreq']
    
    timestamps_frame = pd.read_csv(file_name.removesuffix('edf') + 'csv_bi', header=5)


    bipolar_data = apply_montage(raw_signal_object)
    epochs = mne.make_fixed_length_epochs(bipolar_data, duration=epoch_length) # create epochs
    epoch_tensor = torch.tensor(epochs.get_data())
    annotated_data = write_annotations(bipolar_data, timestamps_frame)
    labels = make_labels(epoch_tensor, timestamps_frame)

    return bipolar_data, annotated_data, epoch_tensor, labels

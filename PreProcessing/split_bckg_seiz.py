"""Split the recordings by background and seizure activity for down/up-sampling

Input: list of patients with seizure activity. 

Preprocessing
1. The signals are filtered with HP 1Hz and LP 70Hz.
2. Notch filter at 60Hz. 
3. Apply bipolar montage


Class partitioning
5. Extract seizure start and stop times from .csv_bi files. Round to the nearest (half/fifth/10th) 
of a second. 
6. Break up recordings according to the class labels. (MNE event based epoching?)

Upsampling
7. Epoch 2 second windows (mne.fixed_length_epochs) with a (0.5) second stride for seizure activity.
8. Resample at 250Hz. 

Evaluation
8. Count total number of windows in each class. 
9. Experiment with different strides to see where we can get the best balance 
(remember we might use more background data) 
 

For each file, we are left with a bunch of snippets containing either background or seizure 
activity. The class labels should be stored along with the signal data. 
"""


import mne 
import pandas as pd
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def remove_suffix(word, suffixes):
    """Remove any suffixes contained in the 'suffixes' array from 'word'"""
    for suffix in suffixes:
        if word.endswith(suffix):
            return word.removesuffix(suffix)
    return word


# with open('Users/toucanfirm/Desktop/train_patients_with_seizures.txt') as file:
#     for patient in tqdm(file): #iterate through patients
#         for edf_file in glob.glob('/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/' + patient, + '/**/*.edf', recursive=True):
#             raw = mne.io.read_raw_edf(edf_file, infer_types=True)
#             raw_notched = raw.copy().notch_filter(freqs=60)
#             raw_filtered = raw_notched.copy().filter(lfreq=1, hfreq=70)

#             channel_renaming_dict = {name: remove_suffix(name, ['-LE', '-REF']) for name in raw.ch_names}
#             raw.rename_channels(channel_renaming_dict)
#             #print(data.ch_names)
#             bipolar_data = mne.set_bipolar_reference(
#             raw.load_data(), 
#             anode=['FP1', 'F7', 'T3', 'T5',
#                 'FP1', 'F3', 'C3', 'P3',
#                 'FP2', 'F4', 'C4', 'P4', 
#                 'FP2', 'F8', 'T4', 'T6',
#                 'T3', 'C3', 'CZ', 'C4'], 
#             cathode=['F7','T3', 'T5', 'O1', 
#                     'F3', 'C3', 'P3', 'O1', 
#                     'F4', 'C4', 'P4', 'O2', 
#                     'F8', 'T4', 'T6', 'O2',
#                     'C3', 'CZ', 'C4', 'T4'], 
#             drop_refs=True)
#             bipolar_data.pick_channels(['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
#                                 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
#                                 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
#                                 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
#                                 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])
            

edf_file = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaaaac/s001_2002_12_23/02_tcp_le/aaaaaaac_s001_t000.edf"

raw = mne.io.read_raw_edf(edf_file, infer_types=True, preload=True)
raw_notched = raw.copy().notch_filter(freqs=60)
raw_filtered = raw_notched.copy().filter(l_freq=1, h_freq=70)

channel_renaming_dict = {name: remove_suffix(name, ['-LE', '-REF']) for name in raw_filtered.ch_names}
raw_filtered.rename_channels(channel_renaming_dict)
#print(data.ch_names)
bipolar_data = mne.set_bipolar_reference(
raw_filtered.load_data(), 
anode=['FP1', 'F7', 'T3', 'T5',
    'FP1', 'F3', 'C3', 'P3',
    'FP2', 'F4', 'C4', 'P4', 
    'FP2', 'F8', 'T4', 'T6',
    'T3', 'C3', 'CZ', 'C4'], 
cathode=['F7','T3', 'T5', 'O1', 
        'F3', 'C3', 'P3', 'O1', 
        'F4', 'C4', 'P4', 'O2', 
        'F8', 'T4', 'T6', 'O2',
        'C3', 'CZ', 'C4', 'T4'], 
drop_refs=True)
bipolar_data.pick_channels(['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                    'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])
            
csv_bi_file = edf_file.removesuffix('.edf') + '.csv_bi'
df = pd.read_csv(csv_bi_file, header=5)
if (df['label'].eq('seiz')).any():
    print(f"Oh yeah there is a seizure in file {edf_file}")
    seizures = []
    background = []
    bckg_start = 0

    bckg_start_times = []
    bckg_end_times = []
    sz_start_times = []
    sz_end_times = []

    for row in df.iterrows():
        sz_start = int((round(row[1]['start_time']*2)/2)*raw.info['sfreq'])
        sz_start_times.append(sz_start)
        sz_stop = int((round(row[1]['stop_time']*2)/2)*raw.info['sfreq'])
        sz_end_times.append(sz_stop)
        seizures.append(bipolar_data._data[:,sz_start:sz_stop])

        if sz_start > 500: #if there is at least two seconds of background at the start of the recording
            background.append(bipolar_data._data[:, bckg_start:sz_start])
            bckg_start_times.append(bckg_start)
        bckg_start = sz_stop

    if bipolar_data._data.shape[1] > (sz_stop + 500): #if there is at least two seconds of bckg at the end of the recording
        background.append(bipolar_data._data[:,sz_stop:])
            


elif (df['label'].eq('bckg')).any():
    print(f"Oh no there is only background noise in file {edf_file}")


plt.figure(0)
plt.subplot(211)
plt.plot(bipolar_data._data[0])
plt.ylabel('Amplitude [microvolts]')
plt.xlabel('Time [seconds/250]')
plt.subplot(212)
plt.plot(np.linspace(0,sz_start_times[0],9250), background[0][0], 'r-', np.linspace(sz_start_times[0],sz_end_times[0],(sz_end_times[0] - sz_start_times[0])), seizures[0][0], 'g', np.linspace(sz_end_times[0],75250,(75250 - sz_end_times[0])), background[1][0], 'r-')
plt.xlabel('Time [seconds/250]')
plt.ylabel('Amplitude [microvolts]')
plt.show()
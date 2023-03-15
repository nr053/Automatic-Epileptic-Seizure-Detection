"""
This script loads the first .edf file from the train set for inspection.
"""

import mne 

path = '/Volumes/KINGSTON/TUSZ_V2/edf/train/aaaaaaac/s001_2002_12_23/02_tcp_le/aaaaaaac_s001_t000.edf'

data = mne.io.read_raw_edf(path, infer_types=True) 
raw_data = data.get_data()

print(data.info)

#data.plot_psd(fmax=125)
data.plot(duration=5, n_channels=33)

for idx, chName in enumerate(data.info['ch_names']):
    print(idx, chName)

print(mne.channels.get_builtin_montages())

# for montage in mne.channels.get_builtin_montages():
#     montage1 = mne.channels.make_standard_montage(montage)
#     print(montage1)
#     montage1.plot()  # 2D

bipolar_data = mne.set_bipolar_reference(
    data.load_data(), 
    anode=['FP1-LE', 'F7-LE', 'T3-LE', 'T5-LE',
         'FP1-LE', 'F3-LE', 'C3-LE', 'P3-LE',
         'FP2-LE', 'F4-LE', 'C4-LE', 'P4-LE', 
         'FP2-LE', 'F8-LE', 'T4-LE', 'T6-LE'], 
    cathode=['F7-LE','T3-LE', 'T5-LE', 'O1-LE', 
             'F3-LE', 'C3-LE', 'P3-LE', 'O1-LE', 
             'F4-LE', 'C4-LE', 'P4-LE', 'O2-LE', 
             'F8-LE', 'T4-LE', 'T6-LE', 'O2-LE'], 
    drop_refs=False)

print(bipolar_data.info)
bipolar_data.plot(duration=5,n_channels=49)
"""An improved version of split_bckg_seiz.py that includes all of the preprocessing steps


1. Filter 1-70Hz
2. Notch filter 60Hz
3. Bipolar montage
4. Resample at 250Hz
5. Crop by class 
6. Epoch classes with appropriate stride
7. Store final result somehow
"""



import mne 
import pandas as pd
from parent import p_tools
from parent.path import Path

file = Path.data + '/edf/train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf'
recording_name = file.split('/')[-1]
#filtering
raw = mne.io.read_raw_edf(file, infer_types=True, preload=True)
raw_filtered = raw.copy().filter(l_freq=1, h_freq=70)
raw_notch = raw_filtered.copy().notch_filter(freqs=60)
#montage
raw_bipolar = p_tools.apply_montage(raw_notch)
#resample
print(f"Original sampling frequency: {raw_bipolar.info['sfreq']}")
if raw_bipolar.info['sfreq'] != 250.0:
    raw_resampled = raw_bipolar.copy().resample(sfreq=250)
    print(f"New sampling frequency: {raw_resampled.info['sfreq']}")
#raw.plot(n_channels=10, duration=5)
#raw_bipolar.plot(n_channels=10, duration=5)
#raw_resampled.plot(n_channels=10, duration=5)

df = pd.read_csv(file.removesuffix('edf')+'csv_bi', header=5)





background_samples = []
seizure_samples = []

# #handling the recording start and end activity
# if (df['label'].eq('seiz')).any():
#     first_row = True
#     for row in df.iterrows():
#         if first_row and row[1]['start_time'] > 2: #if there is at least two seconds of background at the start of the recording
#             bckg_start=0.0 #first background period starts at time=0s   

#         seizure_start_time = row[1]['start_time']
#         seizure_stop_time = row[1]['stop_time']

#         bckg = raw_resampled.copy().crop(tmin=bckg_start, tmax=seizure_start_time) 
#         sz = raw_resampled.copy().crop(tmin=seizure_start_time, tmax=seizure_stop_time)

#         background_times.append({"start_time": bckg_start, "stop_time": bckg_start + len(bckg)/bckg.info['sfreq']})
#         seizure_times.append({"start_time": seizure_start_time, "stop_time": seizure_start_time + len(sz)/sz.info['sfreq']})            

#         background_samples.append(bckg)
#         seizure_samples.append(sz)
        
#         bckg_start = seizure_stop_time #next background period starts at the end of the last seizure

#         first_row = False 

#     if seizure_stop_time+2 < raw_resampled.times[-1]: #if there is at least 2 seconds of background activity at the end of the file
#         bckg = raw_resampled.copy().crop(tmin = bckg_start, tmax = raw_resampled.times[-1])
#         background_samples.append(bckg)
#         background_times.append({"start_time": bckg_start, "stop_time": bckg_start + len(bckg)/bckg.info['sfreq']})


# else: 
#     background_samples.append(raw_resampled())


# assuming all recordings start and end with background activity
# if the recording contains seizures
#   first background recording is from 0.0 - start of first seizure
#   first seizure is from start - end of first seizure
#   crop the recording into background and seizure recordings 
#   appending each recording to a list of corresponding class






if (df['label'].eq('seiz')).any():
    bckg_start=0.0 
    for row in df.iterrows():
        seizure_start_time = row[1]['start_time']
        seizure_stop_time = row[1]['stop_time']

        bckg = raw_resampled.copy().crop(tmin=bckg_start, tmax=seizure_start_time) 
        sz = raw_resampled.copy().crop(tmin=seizure_start_time, tmax=seizure_stop_time)        

        background_samples.append({"recording": bckg, "original_file": recording_name, "original_time_window": (bckg_start,bckg_start + len(bckg)/bckg.info['sfreq']), "cropped":True})
        seizure_samples.append({"recording": sz, "original_file": recording_name, "original_time_window": (seizure_start_time,seizure_start_time + len(sz)/sz.info['sfreq']), "cropped":True})
        
        bckg_start = seizure_stop_time #next background period starts at the end of the last seizure

    #add the last background activity 
    bckg = raw_resampled.copy().crop(tmin = bckg_start, tmax = raw_resampled.times[-1])
    background_samples.append({"recording": bckg, "original_file": recording_name, "original_time_window": (bckg_start, bckg_start + len(bckg)/bckg.info['sfreq']), "cropped":True})


else: 
    background_samples.append({"recording": raw_resampled(), "original_file":recording_name, "original_time_window": None, "cropped": False})

#What happens if a file less than two seconds is epoched? Error is thrown
#drop the background recordings less than 2 seconds

background_epochs = []
seizure_epochs = []
epoch_list = []

for bckg_sample in background_samples:
    if len(bckg_sample['recording'])/bckg_sample['recording'].info['sfreq'] < 2:
        background_samples.remove(bckg_sample)
    else:
        bckg_sample['epochs'] = mne.make_fixed_length_epochs(bckg_sample['recording'], duration=2.0, overlap=0.0) #2 second stride
        epoch_start = 0
        for epoch in bckg_sample['epochs'].get_data():
            epoch_list.append({'epoch': [epoch], 'recording': bckg_sample['original_file'], 'timestamp': bckg_sample['original_time_window'][0]+(epoch_start), 'gt':0, 'pred':None})
            epoch_start+=2

for sz_sample in seizure_samples:
    if len(sz_sample['recording'])/sz_sample['recording'].info['sfreq'] <2:
        seizure_samples.remove(sz_sample)
    else:
        sz_sample['epochs'] = mne.make_fixed_length_epochs(sz_sample['recording'], duration=2.0, overlap=1.5) #0.5 second stride
        epoch_start = 0
        for epoch in sz_sample['epochs'].get_data():
            epoch_list.append({'epoch': [epoch], 'recording': sz_sample['original_file'], 'timestamp': sz_sample['original_time_window'][0]+(epoch_start), 'gt':1, 'pred':None})
            epoch_start+=0.5


data_df = pd.DataFrame(epoch_list)

# number_of_background_epochs = 0
# number_of_seizure_epochs = 0

# for bckg_epoch in background_epochs:
#     number_of_background_epochs += bckg_epoch.get_data().shape[0]

# for sz_epoch in seizure_epochs:
#     number_of_seizure_epochs += sz_epoch.get_data().shape[0]


# print(f"Number of background epochs: {number_of_background_epochs}")
# print(f"Number of seizure epochs after upsampling: {number_of_seizure_epochs}")

#compare downsample then epoch
#with epoch then downsample






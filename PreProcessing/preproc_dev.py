"""Preprocessing of the evaluation and dev sets used for training. 
Very similar procedure to the preproc2.py script used for the training set but without upsampling.

1. Filter 1-70 Hz
2. Notch filter 60 Hz
3. Apply montage
4. Resample
6. Break all samples into 2 second windows ("epochs")


The real life scenario will contain epochs of both seizure and background activity. Since we are training the model on epochs of purely background/seizure
activity, it follows that we should calculate loss and accuracy on windows of the same nature. 

To naively split recordings into two second windows would result in windows that contain mostly background/seizure activity, or an even split. How do we determine
the ground truth label for such a time window? 

Splitting recordings into background/seizure activity ensures that each window has the correct ground truth label. 

Since the optimal model is chosen based on the performance on this dataset it follows that the data should be as "clean" as possible.

The final dataframe only includes time windows of events that are two seconds or longer.



Notes:

Recording: '/home/migo/TUHP/TUSZ_V2/edf/dev/aaaaamnk/s002_2012_06_20/01_tcp_ar/aaaaamnk_s002_t001.edf' threw error: 

"ValueError: tmax (3601.0) must be less than or equal to the max time (3600.9960 sec)"

Changed the final seizure end time to "3600.9960". 

"""


import mne 
import pandas as pd
from parent import p_tools
from parent.path import Path
from tqdm import tqdm 
import glob
import torch 
import logging

logging.basicConfig(filename="preproc_dev_warnings.txt", level=logging.DEBUG)
logging.captureWarnings(True)

epoch_list = []


for file in tqdm(glob.glob(Path.data + '/edf/dev/**/*.edf', recursive=True)):
    recording_name = file.split('/')[-1]
    #filtering
    raw = mne.io.read_raw_edf(file, infer_types=True, preload=True)
    if len(raw)/raw.info['sfreq'] > 5: #recordings shorter than 5 seconds are dropped because they cannot be filtered without distortion
        raw.filter(l_freq=1, h_freq=70)
        raw.notch_filter(freqs=60)
        #montage
        raw_bipolar = p_tools.apply_montage(raw)
        #resample
        #print(f"Original sampling frequency: {raw_bipolar.info['sfreq']}")
        if raw_bipolar.info['sfreq'] != 250.0:
            raw_bipolar.resample(sfreq=250)
            #print(f"New sampling frequency: {raw_bipolar.info['sfreq']}")
        #raw.plot(n_channels=10, duration=5)
        #raw_bipolar.plot(n_channels=10, duration=5)
        #raw_bipolar.plot(n_channels=10, duration=5)

        df = pd.read_csv(file.removesuffix('edf')+'csv_bi', header=5)

        #background_samples = []
        #seizure_samples = []
        cropped_recordings = []

        if (df['label'].eq('seiz')).any():
            bckg_start=0.0 
            for row in df.iterrows():

                seizure_start_time = row[1]['start_time']
                seizure_stop_time = row[1]['stop_time']

                if seizure_stop_time > raw_bipolar.times[-1]:
                    seizure_stop_time = raw_bipolar.times[-1]

                bckg = raw_bipolar.copy().crop(tmin=bckg_start, tmax=seizure_start_time) 
                sz = raw_bipolar.copy().crop(tmin=seizure_start_time, tmax=seizure_stop_time)        

                #background_samples.append({"recording": bckg, "original_file": file, "original_time_window": (bckg_start,bckg_start + len(bckg)/bckg.info['sfreq']), "cropped":True})
                #seizure_samples.append({"recording": sz, "original_file": file, "original_time_window": (seizure_start_time,seizure_start_time + len(sz)/sz.info['sfreq']), "cropped":True})
                
                cropped_recordings.append({"recording": bckg, "original_file": file, "original_time_window": (bckg_start,bckg_start + len(bckg)/bckg.info['sfreq']), "gt":0})
                cropped_recordings.append({"recording": sz, "original_file": file, "original_time_window": (seizure_start_time,seizure_start_time + len(sz)/sz.info['sfreq']), "gt":1})


                bckg_start = seizure_stop_time #next background period starts at the end of the last seizure

            #add the last background activity 
            bckg = raw_bipolar.copy().crop(tmin = bckg_start, tmax = raw_bipolar.times[-1])
            #background_samples.append({"recording": bckg, "original_file": file, "original_time_window": (bckg_start, bckg_start + len(bckg)/bckg.info['sfreq']), "cropped":True})
            cropped_recordings.append({"recording": bckg, "original_file": file, "original_time_window": (bckg_start,bckg_start + len(bckg)/bckg.info['sfreq']), "gt":0})

        #else: 
        #    background_samples.append({"recording": raw_bipolar, "original_file":recording_name, "original_time_window": (0,len(raw_bipolar)/raw_bipolar.info['sfreq']), "cropped": False})


        # for bckg_sample in background_samples:
        #     if len(bckg_sample['recording'])/bckg_sample['recording'].info['sfreq'] > 2:
        #         bckg_sample['epochs'] = mne.make_fixed_length_epochs(bckg_sample['recording'], duration=2.0, overlap=0.0) #2 second stride
        #         epoch_start = 0
        #         epoch_index = 0
        #         for epoch in bckg_sample['epochs'].get_data():
        #             epoch_tensor = torch.Tensor(epoch)
        #             epoch_path = '/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/DevEpochs/Data_only_records_with_seizures/' + bckg_sample['original_file'].split('/')[-1].removesuffix('.edf') + '_' + str(epoch_index) + '.pt'
        #             torch.save(epoch_tensor, epoch_path)
        #             epoch_list.append({'epoch': epoch_path, 'recording': bckg_sample['original_file'], 'timestamp': bckg_sample['original_time_window'][0]+(epoch_start), 'gt':0, 'pred':None})
        #             epoch_start+=2
        #             epoch_index+=1


        # for sz_sample in seizure_samples:
        #     if len(sz_sample['recording'])/sz_sample['recording'].info['sfreq'] > 2:
        #         sz_sample['epochs'] = mne.make_fixed_length_epochs(sz_sample['recording'], duration=2.0, overlap=0) #2 second stride
        #         epoch_start = 0
        #         epoch_index = 0
        #         for epoch in sz_sample['epochs'].get_data():
        #             epoch_tensor = torch.Tensor(epoch)
        #             epoch_path = '/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/DevEpochs/Data_only_records_with_seizures/' + sz_sample['original_file'].split('/')[-1].removesuffix('.edf') + '_' + str(epoch_index) + '.pt'
        #             torch.save(epoch_tensor, epoch_path)
        #             epoch_list.append({'epoch': epoch_path, 'recording': sz_sample['original_file'], 'timestamp': sz_sample['original_time_window'][0]+(epoch_start), 'gt':1, 'pred':None})
        #             epoch_start+=2
        #             epoch_index+=1

        previous_recording = None

        for recording in cropped_recordings:
            if len(recording['recording'])/recording['recording'].info['sfreq'] > 2:
                print("Epoching recording")
                epochs = mne.make_fixed_length_epochs(recording['recording'], duration=2.0, overlap=0.0) #2 second stride for all windows. 

                epoch_start = recording['original_time_window'][0]

                for epoch in epochs.get_data():

                    if recording['original_file'].split('/')[-1] == previous_recording:
                        epoch_index += 1
                    else:
                        epoch_index = 0
                    
                    epoch_tensor = torch.Tensor(epoch)
                    epoch_path = '/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/DevEpochs/Data/' + recording['original_file'].split('/')[-1].removesuffix('.edf') + '_' + str(epoch_index) + '.pt'
                    torch.save(epoch_tensor, epoch_path)
                    
                    epoch_list.append({'epoch': epoch_path, 'recording': recording['original_file'], 'timestamp': epoch_start, 'gt':recording['gt'], 'pred': None})

                    previous_recording = recording['original_file'].split('/')[-1]
                    epoch_start+=2


df_dev = pd.DataFrame(epoch_list)
df_dev.to_csv(Path.repo + '/DevEpochs/dev_only_records_with_seizures.csv')



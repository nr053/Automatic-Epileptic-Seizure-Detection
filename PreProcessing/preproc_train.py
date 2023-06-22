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
from tqdm import tqdm 
import glob
import torch 
import time

epoch_list = []

with open(Path.repo + '/TrainingEpochs/train_patients_with_seizures.txt', 'r') as patient_file:
    for patient in tqdm(patient_file):
        #print(patient.rstrip())
        for file in glob.glob(Path.data + '/edf/train/' + patient.rstrip() + '/**/*.edf', recursive=True):

            recording_name = file.split('/')[-1]
            #filtering
            raw = mne.io.read_raw_edf(file, infer_types=True, preload=True)
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

            #What happens if a file less than two seconds is epoched? Error is thrown
            #drop the background recordings less than 2 seconds

            # for bckg_sample in background_samples:
            #     if len(bckg_sample['recording'])/bckg_sample['recording'].info['sfreq'] > 2:
            #         bckg_sample['epochs'] = mne.make_fixed_length_epochs(bckg_sample['recording'], duration=2.0, overlap=0.0) #2 second stride
            #         epoch_start = 0
            #         epoch_index = 0
            #         for epoch in bckg_sample['epochs'].get_data():
            #             epoch_tensor = torch.Tensor(epoch)
            #             epoch_path = '/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/TrainingEpochs/Data/' + bckg_sample['original_file'].split('/')[-1].removesuffix('.edf') + '_' + str(epoch_index) + '.pt'
            #             torch.save(epoch_tensor, epoch_path)
            #             epoch_list.append({'epoch': epoch_path, 'recording': bckg_sample['original_file'], 'timestamp': bckg_sample['original_time_window'][0]+(epoch_start), 'gt':0, 'pred':None})
            #             epoch_start+=2
            #             epoch_index+=1


            # for sz_sample in seizure_samples:
            #     if len(sz_sample['recording'])/sz_sample['recording'].info['sfreq'] > 2:
            #         sz_sample['epochs'] = mne.make_fixed_length_epochs(sz_sample['recording'], duration=2.0, overlap=1.5) #0.5 second stride
            #         epoch_start = 0
            #         epoch_index = 0
            #         for epoch in sz_sample['epochs'].get_data():
            #             epoch_tensor = torch.Tensor(epoch)
            #             epoch_path = '/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/TrainingEpochs/Data/' + sz_sample['original_file'].split('/')[-1].removesuffix('.edf') + '_' + str(epoch_index) + '.pt'
            #             torch.save(epoch_tensor, epoch_path)
            #             epoch_list.append({'epoch': epoch_path, 'recording': sz_sample['original_file'], 'timestamp': sz_sample['original_time_window'][0]+(epoch_start), 'gt':1, 'pred':None})
            #             epoch_start+=0.5
            #             epoch_index+=1

            previous_recording = None

            print("")
            print("Iterating through the cropped recordings")
            print("")
            

            for recording in cropped_recordings:
                if len(recording['recording'])/recording['recording'].info['sfreq'] > 2:
                    print("Epoching recording")
                    if recording['gt'] == 0:
                        epochs = mne.make_fixed_length_epochs(recording['recording'], duration=2.0,overlap=0.0) #no overlap for background windows
                    elif recording['gt'] == 1:
                        epochs = mne.make_fixed_length_epochs(recording['recording'], duration=2.0, overlap=1.5) #0.5 second stride for seizure windows. 

                    epoch_start = recording['original_time_window'][0]

                    print("")
                    print("--iterating through the epochs--")

                    for epoch in epochs.get_data():
                        st = time.time()

                        if recording['original_file'].split('/')[-1] == previous_recording:
                            epoch_index += 1
                        else:
                            epoch_index = 0

                        et = time.time()

                        print("")
                        print(f"Incrementing epoch_index took: {et-st}")


                        st = time.time()
                        
                        epoch_tensor = torch.Tensor(epoch)
                        epoch_path = '/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/TrainingEpochs/Data/' + recording['original_file'].split('/')[-1].removesuffix('.edf') + '_' + str(epoch_index) + '.pt'
                        torch.save(epoch_tensor, epoch_path)

                        et = time.time()
                        print(f"Saving took: {et - st}")
                        st = time.time()
                        
                        epoch_list.append({'epoch': epoch_path, 'recording': recording['original_file'], 'timestamp': epoch_start, 'gt':recording['gt'], 'pred': None})
                        
                        et = time.time()
                        print(f"appending took: {et-st}")

                        previous_recording = recording['original_file'].split('/')[-1]

                        st = time.time()

                        if recording['gt'] == 0:
                            epoch_start+=2
                        elif recording['gt'] == 1:
                            epoch_start+=0.5
                        
                        et = time.time()

                        print(f"Incrementing epoch start time took: {et-st}")

patient_file.close()
data_df = pd.DataFrame(epoch_list)
data_df.to_csv('/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/TrainingEpochs/train_only_records_with_seizures.csv')





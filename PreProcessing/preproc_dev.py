"""Preprocessing of the evaluation and dev sets used for training. 
Very similar procedure to the preproc2.py script used for the training set but without upsampling and filtering of records.

1. Filter 1-70 Hz
2. Notch filter 60 Hz
3. Apply montage
4. Resample
6. Break all samples into 2 second windows ("epochs")


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

        seizure_times = []

        if (df['label'].eq('seiz')).any():
            for row in df.iterrows():
                seizure_times.append((round(row[1]['start_time'] * 2)/2, round(row[1]['stop_time']*2)/2)) #round times to nearest half second

        epoch_windows = mne.make_fixed_length_epochs(raw_bipolar, duration=2.0, overlap=1.5) #2 second windows with a half second stride
        epoch_start = 0
        epoch_index = 0
        for epoch in epoch_windows.get_data():
            epoch_tensor = torch.Tensor(epoch)
            epoch_path = Path.repo + '/DevEpochs/Data/' + recording_name.removesuffix('.edf') + '_' + str(epoch_index) + '.pt' 
            torch.save(epoch_tensor, epoch_path)
            ground_truth = 0 #background label by default
            for times in seizure_times:
                if times[0] <= epoch_start < times[1]:
                    ground_truth = 1

            epoch_list.append({'epoch': epoch_path, 'recording': file, 'timestamp': epoch_start, 'gt': ground_truth, 'pred': None})
            epoch_start+=0.5
            epoch_index+=1

df_dev = pd.DataFrame(epoch_list)
df_dev.to_csv(Path.repo + '/DevEpochs/dev.csv')





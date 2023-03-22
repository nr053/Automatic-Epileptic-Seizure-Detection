"""Epoch continuous data and visualize."""

import mne
import pandas as pd

def remove_suffix(word, suffixes):
    """Remove any suffixes contained in the 'suffixes' array from 'word'"""
    for suffix in suffixes:
        if word.endswith(suffix):
            return word.removesuffix(suffix)
    return word

######## LOAD FILE AND READ DATA ############

#file = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaartn/s007_2014_11_09/03_tcp_ar_a/aaaaartn_s007_t019.edf"
#file = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaartn/s007_2014_11_09/03_tcp_ar_a/aaaaartn_s007_t018.edf"
file = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf"
data = mne.io.read_raw_edf(file, infer_types=True)
duration = data._raw_extras[0]['n_records']
#print("Recording duration = ", duration)
#print(data.info)

############# APPLY BIPOLAR MONTAGE #############

channel_renaming_dict = {name: remove_suffix(name, ['-LE', '-REF']) for name in data.ch_names}
data.rename_channels(channel_renaming_dict)
#print(data.ch_names)
bipolar_data = mne.set_bipolar_reference(
    data.load_data(), 
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

#print("CHANNNNAMES::: ", bipolar_data.ch_names)
#print(bipolar_data.info)


########### READ/WRITE ANNOTATIONS ##################

df = pd.read_csv(file.removesuffix('edf') + 'csv_bi', header=5)
onset_times = df['start_time'].values
durations = df['stop_time'].values - df['start_time'].values 
description = ["seizure"]
annotations = mne.Annotations(onset_times, durations, description)
bipolar_data.set_annotations(annotations)


############ VISUALIZE EPOCHS #####################

bipolar_data.plot(duration=5, n_channels=20, order=list(range(17,37)))

#events = mne.make_fixed_length_events(data, duration=1.0)
#epochs = mne.Epochs(data, events, tmin=0, tmax=0, baseline=None)
epochs = mne.make_fixed_length_epochs(bipolar_data, duration=1)
epochs.plot(n_epochs=5, picks=['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                                                   'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                                                   'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                                                   'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                                                   'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])
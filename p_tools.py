"""
This is the prep_tools module, containing methods used to prepare raw edf files
for input to a deep learning model including:
    
1. removing suffixes from channel names ("-LE", "-REF")
2. creating class labels
3. Calculating features
"""

import torch
from scipy.stats import kurtosis
from scipy.fft import fft
import mne
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_name, epoch_length):
    raw_signal_object = mne.io.read_raw_edf(file_name, infer_types=True)
    timestamps_frame = pd.read_csv(file_name.removesuffix('edf') + 'csv_bi', header=5)

    bipolar_data = apply_montage(raw_signal_object)
    epochs = mne.make_fixed_length_epochs(bipolar_data, duration=epoch_length) # create epochs
    epoch_tensor = torch.tensor(epochs.get_data())
    annotated_data = write_annotations(bipolar_data, timestamps_frame)
    labels = make_labels(epoch_tensor, timestamps_frame)

    return bipolar_data, annotated_data, epoch_tensor, labels

def remove_suffix(word, suffixes):
    """Remove any suffixes contained in the 'suffixes' array from 'word'"""
    for suffix in suffixes:
        if word.endswith(suffix):
            return word.removesuffix(suffix)
    return word

def make_labels(epoch_tensor, df):
    """Creates class labels for the input data using the csv file"""
    labels = torch.empty([epoch_tensor.shape[0],2])
    for i,epoch in enumerate(labels):
        labels[i] = torch.tensor([1,0]) #set all class labels to [1,0] (background activity)
    for row in df.iterrows():
        if row[1]['label'] == "seiz":
            labels[round(row[1]['start_time']):round(row[1]['stop_time'])] = torch.tensor([0,1]) #set seizure class labels (round to nearest second)
    return labels

def calc_features(epoch_tensor):
    """This function calculates representative features of each channel in an epoch. 
    Each epoch is represented by 20 channels, which are represented by a list of features:
    
    1. Epoch mean
    2. Epoch variance
    3. Kurtosis
    4. """

    size = list(epoch_tensor.shape[:2])
    size.append(3)
    features = torch.empty(size)
    for i, epoch in enumerate(epoch_tensor):
        features[i] = torch.stack([epoch.mean(dim=1), epoch.var(dim=1), torch.tensor(kurtosis(epoch.numpy(),axis=1))],dim=1)
    return features

def apply_montage(data):
    """Apply the bipolar montage"""
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
    bipolar_data.pick_channels(['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                            'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                            'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])
    return bipolar_data

def write_annotations(bipolar_data, labels_df):
    """Read annotations from csv file and write them to the mne object"""
    if labels_df['label'].eq("seiz").any(): #if the event df contains a seizure
        onset_times = labels_df['start_time'].values
        durations = labels_df['stop_time'].values - labels_df['start_time'].values 
        description = ["seizure"]
        annotations = mne.Annotations(onset_times, durations, description)
        bipolar_data.set_annotations(annotations)
    return bipolar_data

def plot_spectrogram_plt(bipolar_data):
    """Plot the spectrogram using matplotlib"""
    data_plot = bipolar_data.get_data()[0]
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data_plot, Fs=256)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    plt.subplot(2,1,1)
    plt.specgram(data_plot[650:], 256)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(2,1,2)
    plt.specgram(data_plot[540:640], 256)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()





# with open('file_list_256.txt') as f:
#     lines = [line.rstrip() for line in f]


# for i in range(5):
#     signal_obj, timestamps_df = load_data(lines[i])
#     bipolar = write_annotations(apply_montage(signal_obj), timestamps_df)
#     epochs = mne.make_fixed_length_epochs(bipolar, duration=1)
#     epoch_tensor = torch.tensor(epochs.get_data())
#     labels = make_labels(epoch_tensor, timestamps_df)
#     bipolar.plot(duration=5,highpass=1, lowpass=70, n_channels=20)

#     print("THIS IS SIGNAL NUMBER: ", i)
#     print(timestamps_df)

#     for j in range(len(labels)):
#         if labels[j] == [0,1]:
#             print(j)
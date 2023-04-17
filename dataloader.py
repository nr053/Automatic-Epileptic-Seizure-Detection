import os
import torch
import pandas as pd
import mne
from torch.utils.data import Dataset, DataLoader
from path import Path
from p_tools import apply_montage, remove_suffix, write_annotations, load_data, make_labels
import glob
import matplotlib.pyplot as plt


class EEG_Dataset(Dataset):
    """EEG Dataset class for the raw signal inputs."""

    def __init__(self, file_list, epoch_duration, transform=None):
        """
        Arguments: 
            directory (string): Directory containing all the .edf files. 
            file_list (list(str)): List of files to be used. 
            epoch_dur (float): Duration of the fixed length epochs (seconds).
        """
        self.file_list = file_list
        self.transform = transform
        self.epoch_dur = epoch_duration

        with open(self.file_list) as f:
            self.lines = [line.rstrip() for line in f]





    def __len__(self):
        return len(self.lines)
        
    def __getitem__(self, idx):
        edf_object, label_df = load_data(self.lines[idx])
        bipolar_data = apply_montage(edf_object)
        bipolar_data = write_annotations(bipolar_data, label_df)

        epochs = mne.make_fixed_length_epochs(bipolar_data, duration=self.epoch_dur) # create epochs
        epoch_tensor = torch.tensor(epochs.get_data())
        labels = make_labels(epoch_tensor, label_df)

        sample = {'raw_signal': epoch_tensor, 'labels': labels}

        return bipolar_data, sample



        
dataset = EEG_Dataset(file_list='file_list_256.txt', epoch_duration=1)

fig = plt.figure()

for i in range(4):
    bipolar_data, sample = dataset[i]

    print(i, sample['raw_signal'].shape, sample['labels'].shape)
    bipolar_data.plot(duration=5,highpass=1, lowpass=70, n_channels=20)
    
plt.show()
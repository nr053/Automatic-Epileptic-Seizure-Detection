"""
Dataloader class for CNN architecture. Upon initialisation the edf files and corresponding csv files are read and loaded into a tensor of input:X and labels:y
"""


import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class CNN_Dataset(Dataset):
    """EEG Dataset class for the raw signal inputs."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments: 
            csv_file: path to the csv file containing paths of epoch tensors and labels
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform




    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        epoch_path = self.df.iloc[idx]['epoch']
        epoch_tens = torch.load(epoch_path).double()
        epoch_tens = epoch_tens[None,:]

        label = float(self.df.iloc[idx]['gt'])

        sample = {'X': epoch_tens, 'y': label}

        return sample



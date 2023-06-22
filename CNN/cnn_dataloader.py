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
        #self.df = self.dataframe[:10000]
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        epoch_path = self.df.iloc[idx]['epoch']
        epoch_tens = torch.load(epoch_path).double()
        epoch_tens = epoch_tens[None,:]

        label = torch.tensor([1-float(self.df.iloc[idx]['gt']), float(self.df.iloc[idx]['gt'])])

        sample = {'X': epoch_tens, 'y': label, 'idx': idx, 'prediction': -1}

        return sample

    def add_prediction(self, sample):
        idx_pred_list = list(zip(sample['idx'], sample['prediction'].detach()))

        for idx, prediction in idx_pred_list:
            self.df.loc[idx.item(),'pred'] = prediction[1].item()



# class CNN_Dataset(Dataset):
#     """EEG Dataset class for the raw signal inputs."""

#     def __init__(self, csv_file, transform=None):
#         """
#         Arguments: 
#             csv_file: path to the csv file containing paths of epoch tensors and labels
#         """
#         self.df = pd.read_csv(csv_file)
#         #self.df = self.dataframe[:10000]
#         self.transform = transform
#         self.df.loc[:, 'epoch_tensor'] = self.df.apply(lambda row: torch.load(row['epoch']),axis=1)
        
#     def __len__(self):
#         return len(self.df)
        
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         epoch_tens = self.df.loc[idx, 'epoch_tensor']

#         label = torch.tensor([1-float(self.df.iloc[idx]['gt']), float(self.df.iloc[idx]['gt'])])

#         sample = {'X': epoch_tens, 'y': label, 'idx': idx, 'prediction': -1}

#         return sample

#     def add_prediction(self, sample):
#         idx_pred_list = list(zip(sample['idx'], sample['prediction'].detach()))

#         for idx, prediction in idx_pred_list:
#             self.df.loc[idx.item(),'pred'] = prediction[1].item()



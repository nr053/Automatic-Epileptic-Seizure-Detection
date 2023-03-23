"""Epoch continuous data and visualize."""

import mne
import pandas as pd
import tensorflow as tf
import torch
from torch import nn
import numpy as np

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


############ VISUALIZE + EPOCHS #####################

bipolar_data.pick_channels(['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                            'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                            'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])

bipolar_data.plot(duration=5,highpass=1, lowpass=70, n_channels=20)

#events = mne.make_fixed_length_events(data, duration=1.0)
#epochs = mne.Epochs(data, events, tmin=0, tmax=0, baseline=None)
epochs = mne.make_fixed_length_epochs(bipolar_data, duration=1)
epochs.plot(n_epochs=5)


######### DEFINE DL MODEL ARCHITCTURE  #########

"""
Tenemos:
epoched raw data to access. 793 (epochs) x 37 (channels) x 256 (values). Each epoch is treated
as an individual sample, so 793 samples of dimension 20 (useful channels) x 256 values.
Input to the model is a 20 x 256 matrix. 

Raw data is taken from the epochs, and the class value is taken from...

Could use the annotation start and duration times...
Events?

Necesitamos:
tensor to input to tensor flow. 
"""

############################### LAST THING ON THURSDAY #######################
#input = tf.convert_to_tensor(epochs.get_data(), dtype=float)
input = torch.tensor(epochs.get_data())
labels = np.zeros([input.shape[0],2])
#labels = torch.zeros(input.shape[0])
for i in range(input.shape[0]):
    labels[i] =  [1,0]

for row in df.iterrows():
    labels[round(row[1]['start_time']):round(row[1]['stop_time'])] = [0,1]


############################## CHECK THIS WORKED BEFORE MOVING ON. PRETTY SURE IT DID

torch.set_default_dtype(torch.float64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(0,-1)
        self.linear_stack = nn.Sequential(
            nn.Linear(20*256, 3000),
            nn.ReLU(),
            nn.Linear(3000,3000),
            nn.ReLU(),
            nn.Linear(3000,2000),
            nn.ReLU(),
            nn.Linear(2000,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,2)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
    
model = NeuralNetwork()

sample = input[0:10]     # sample the first 10 epochs from the recording
logits = model(sample)

label = torch.tensor([1-labels[0], labels[0

softmax  = nn.Softmax(dim=0)
pred_probab = softmax(logits)
pred_class = torch.round(pred_probab)

####### HYPERPARAMETERS

learning_rate = 1e-3  # rate at which to update the parameters
batch_size = 1        # number of samples used before updating parameters
epochs = 1            # number of iterations over dataset
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss = loss_fn(pred_class, label)

##### BACKPROPOGATION #######

optimizer.zero_grad()
loss.backward()
optimizer.step()
"""
The first model, taking two features, trained and evaluated
"""

import mne
import pandas as pd
import tensorflow as tf
import torch
from torch import nn
import numpy as np
import tqdm
import copy

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
        labels[round(row[1]['start_time']):round(row[1]['stop_time'])] = torch.tensor([0,1]) #set seizure class labels (round to nearest second)
    return labels

def calc_features(epoch_tensor):
    """This function calculates representative features of each channel in an epoch. 
    Each epoch is represented by 20 channels, which are represented by a list of features"""
    epoch_means = torch.empty(epoch_tensor.shape[:2]) #create empty tensor of size = n_channels x n_epochs
    epoch_variance = torch.empty(epoch_tensor.shape[:2]) 
    for i,epoch in enumerate(epoch_tensor):
        epoch_means[i] = epoch.mean(dim=1) # add means to epoch_mean_list
        epoch_variance[i] = epoch.var(dim=1)
    return epoch_means, epoch_variance

torch.set_default_dtype(torch.float64) # set default type of torch.tensors
class NeuralNetwork(nn.Module):
    """Neural network taking two features (mean,variance) as input and returning class labels"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(0,-1)
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4,2)
        )
    
    def forward(self,x):
        #x = self.flatten(x)
        logits = self.linear_stack(x)
        #probs = (softmax(logits))
        #pred = torch.round(probs)
        return logits


file = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf"
data = mne.io.read_raw_edf(file, infer_types=True) # read edf file
channel_renaming_dict = {name: remove_suffix(name, ['-LE', '-REF']) for name in data.ch_names} # define renaming dict
data.rename_channels(channel_renaming_dict) # rename the channels
bipolar_data = mne.set_bipolar_reference( # set bipolar reference montage
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

bipolar_data.pick_channels(['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', # select channels to use
                            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                            'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                            'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])

epochs = mne.make_fixed_length_epochs(bipolar_data, duration=1) # create epochs
epoch_tensor = torch.tensor(epochs.get_data())
df = pd.read_csv(file.removesuffix('edf') + 'csv_bi', header=5) # create dataframe from csv file
labels = make_labels(epoch_tensor, df) # set class labels
epoch_means, epoch_variance = calc_features(epoch_tensor)

model = NeuralNetwork()

X_train = torch.stack((epoch_means[:500,0], epoch_variance[:500,0]),1)      # sample the first epoch from the recording (using only one channel)
X_test = torch.stack((epoch_means[500:,0], epoch_variance[500:,0]),1)     
y_train = labels[:500]
y_test = labels[500:]

learning_rate = 0.001  # rate at which to update the parameters
batch_size = 5        # number of samples used before updating parameters
n_epochs = 200            # number of iterations over dataset
batches_per_epoch = len(X_train) // batch_size 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# def train_loop(input, labels, model, loss_fn, optimizer):
#     size = len(input)
#     #for X,y in input,labels:
#     for i in range(size):
#         #print("THIS IS THE INPUT SIZE: ", input[i].shape)
#         pred = model(input[i])
#         loss = loss_fn(pred,labels[i])
#         #Backpropogation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()  #update the weights
#         print(f"loss: {loss:>7f}")


# num_epochs = 10
# # for i in range(num_epochs):
# #     print(f" Epoch {i+1}: ")
# #     train_loop(sample1[None,:], label1[None,:], model, loss_fn, optimizer)
# # print("Done!")


# for i in range(num_epochs):
#     print(f" Epoch {i+1}: ")
#     train_loop(sample10, label10, model, loss_fn, optimizer)
# print("Done!")

best_acc = - np.inf
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

model.train()  # set the model to training mode (good practice)

for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:    
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store epoch metrics
            acc = (torch.argmax(y_pred,1) == torch.argmax(y_batch,1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(loss=float(loss),accuracy=float(acc))

    model.eval() #put the model in evaluation mode, vital when network contains dropout/normalisation layers
    y_pred = model(X_test)
    loss_test = float(loss_fn(y_pred, y_test))
    accuracy = float((torch.argmax(y_pred,1) == torch.argmax(y_test,1)).float().mean())
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(loss_test)
    test_acc_hist.append(accuracy)
    if accuracy > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: loss={loss}, accuracy={accuracy}")

model.load_state_dict(best_weights)
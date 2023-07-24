"""
Dataloader class. Upon initialisation the edf files and corresponding csv files are read and loaded into a tensor of input:X and labels:y

- .edf files are read
- raw signal values extracted
- signal is split into epochs
- .csv_bi is read 
- labels are created for each epoch


REMEMBER TO ADD LOWPASS, HIGHPASS and NOTCH FILTERS
"""


import os
import torch
import pandas as pd
import mne
from torch.utils.data import Dataset, DataLoader, random_split
from path import Path
from p_tools import apply_montage, remove_suffix, write_annotations, load_data, make_labels
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
import numpy as np
#from sklear.model_selection import StratifiedShuffleSplit
import time 
from progress.bar import Bar


torch.set_default_dtype(torch.float64)

st = time.time()
# class Sampler(object):
#     """Base class for all Samplers.
#     Every Sampler subclass has to provide an __iter__ method, providing a way
#     to iterate over indices of dataset elements, and a __len__ method that
#     returns the length of the returned iterators.
#     """

#     def __init__(self, data_source):
#         pass

#     def __iter__(self):
#         raise NotImplementedError

#     def __len__(self):
#         raise NotImplementedError

# class StratifiedSampler(Sampler):
#     """Stratified Sampling
#     Provides equal representation of target classes in each batch
#     """
#     def __init__(self, class_vector, batch_size):
#         """
#         Arguments
#         ---------
#         class_vector : torch tensor
#             a vector of class labels
#         batch_size : integer
#             batch_size
#         """
#         self.n_splits = int(class_vector.size(0) / batch_size)
#         self.class_vector = class_vector

#     def gen_sample_array(self):
#         try:
#             from sklearn.model_selection import StratifiedShuffleSplit
#         except:
#             print('Need scikit-learn for this functionality')
#         import numpy as np
        
#         s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
#         X = torch.randn(self.class_vector.size(0),2).numpy()
#         y = self.class_vector.numpy()
#         s.get_n_splits(X, y)

#         train_index, test_index = next(s.split(X, y))
#         return np.hstack([train_index, test_index])

#     def __iter__(self):
#         return iter(self.gen_sample_array())

#     def __len__(self):
#         return len(self.class_vector)


class EEG_Dataset(Dataset):
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
        
        epoch_path = self.df.iloc[idx,1]
        epoch_tens = torch.load(epoch_path)

        label = float(self.df.iloc[idx,2])

        sample = {'X': epoch_tens, 'y': label}

        return sample



        
dataset = EEG_Dataset(csv_file='epoch_data_small.csv')
train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size,test_size])




# train_seizures=0
# test_seizures=0
# for sample in train_data:
#     train_seizures += sample['y']
# for i in test_data:
#     test_seizures += sample['y']

# print(f"Number of seizure epochs in training set: {train_seizures}")
# print(f"Number of seizure epochs in test set: {test_seizures}")

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
# for i, batch in enumerate(train_dataloader):
#     print(i, batch)
#     if i ==1:
#         break


class PrintSize(nn.Module):
    """Utility to print the size of the tensor in the current step (only on the first forward pass)"""
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten(0, -1)
        self.convolutional_stack = nn.Sequential(
            #nn.LayerNorm(normalized_shape = [256]),
            PrintSize(),
            nn.Conv1d(in_channels=20, out_channels=40, kernel_size=5, padding=2),
            nn.ReLU(),
            PrintSize(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            PrintSize(),
            nn.Conv1d(in_channels=40,out_channels=80, kernel_size=5, padding=2),
            PrintSize(),
            nn.ReLU(),
            nn.Conv1d(in_channels=80,out_channels=40, kernel_size=5, padding=2),
            PrintSize(),
            nn.ReLU(),
            nn.Flatten(1,-1), #keep the first dimension (batch size) and flatten the rest
            PrintSize(),
            nn.Linear(in_features=3400, out_features=500),
            PrintSize(),
            nn.ReLU(),
            nn.Linear(in_features=500,out_features=1),
            PrintSize(),
            nn.Sigmoid()
        )
    def forward(self,x):
        logits = self.convolutional_stack(x)
        #probs = softmax(logits)
        #preds = torch.round(probs)
        return logits


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten(0, -1)
        self.convolutional_stack = nn.Sequential(
            #nn.LayerNorm(normalized_shape = [256]),
            PrintSize(),
            nn.Conv1d(in_channels=20, out_channels=30, kernel_size=5, padding=2),
            nn.ReLU(),
            PrintSize(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            PrintSize(),
            nn.Conv1d(in_channels=30,out_channels=40, kernel_size=5, padding=2),
            PrintSize(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Flatten(1, -1),
            PrintSize(),
            nn.Linear(in_features=1120, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        logits = self.convolutional_stack(x)
        #probs = softmax(logits)
        #preds = torch.round(probs)
        return logits

model = NeuralNetwork()
model2 = NeuralNetwork2()

learning_rate = 1e-4  # rate at which to update the parameters
n_epochs = 1            # number of iterations over dataset
batch_size = 128
#batches_per_epoch = len(X_train) // batch_size
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# best_acc = - np.inf
# best_weights = None
# train_loss_hist = []
# train_acc_hist = []


def train_loop(dataloader, model, loss_fn, optimizer):
    train_loss_hist = []
    train_acc_hist = []
    model.train()
    size = len(dataloader.dataset)


    #for batch, sample in tqdm(enumerate(dataloader)):
    #with Bar('Processing...') as bar:
    loop = tqdm(dataloader)
    for batch, sample in enumerate(loop):
        pred = model(sample['X'])
        labels = sample['y']
        loss = loss_fn(pred, labels[:,None])

        train_loss_hist.append(loss)
        #print("Calculating accuracy")
        accuracy = (labels == pred).float().mean().item()
        #print("Appending accuracy")
        train_acc_hist.append(accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #    bar.next()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(sample['X'])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #print(f"Pred = {pred}")

    return train_loss_hist, train_acc_hist

def test_loop(dataloader, model, loss_fn):
    test_loss_hist = []
    test_acc_hist = []
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct  = 0,0

    with torch.no_grad():
        for sample in dataloader:
            pred = model(sample['X'])
            labels = sample['y']
            #print(f"Labels shape : {labels.shape}")
            #print(f"Pred shape: {pred.shape}")
            #print(f"Transformed labels shape: {labels[:,None].shape}")
            test_loss += loss_fn(pred, labels[:,None]).item()
            test_loss_hist.append(test_loss)
            #print(f"Shape of prediction: {pred.shape}")
            #print(f"Shape of target: {sample['y'].shape}")
            correct += (pred == labels).type(torch.float).sum().item()
            test_acc_hist.append(correct)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss_hist, test_acc_hist



#train_loop(train_dataloader, model2, loss_fn, optimizer)
#test_loop(test_dataloader, model2, loss_fn)
train_loss_hist, train_acc_hist = train_loop(train_dataloader, model2, loss_fn, optimizer)
test_loss_hist, test_acc_hist = test_loop(test_dataloader, model2, loss_fn)




#Plot the loss and accuracy
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.show()
 
#plt.plot(train_acc_hist, label="train")
plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

et = time.time()

print(f"Execution time: {et-st} seconds")


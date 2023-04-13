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
import matplotlib.pyplot as plt
from p_tools import remove_suffix, make_labels, calc_features, apply_montage
from path import Path


torch.set_default_dtype(torch.float64) # set default type of torch.tensors

class PrintSize(nn.Module):
    first = True

    def forward(self,x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x

class NeuralNetwork(nn.Module):
    """Neural network taking two features (mean,variance) as input and returning class labels"""
    def __init__(self):
        super().__init__()
        PrintSize()
        self.linear_stack = nn.Sequential(
            PrintSize(),
            nn.Flatten(),
            PrintSize(),
            nn.Linear(60, 200),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,2)
        )
    
    def forward(self,x):
        logits = self.linear_stack(x)
        #probs = (softmax(logits))
        #pred = torch.round(probs)
        return logits


file = Path.machine_path + "train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf"
data = mne.io.read_raw_edf(file, infer_types=True) # read edf file

bipolar_data = apply_montage(data)

epochs = mne.make_fixed_length_epochs(bipolar_data, duration=1) # create epochs
epoch_tensor = torch.tensor(epochs.get_data())
df = pd.read_csv(file.removesuffix('edf') + 'csv_bi', header=5) # create dataframe from csv file
labels = make_labels(epoch_tensor, df) # set class labels
features = calc_features(epoch_tensor)

model = NeuralNetwork()

X_train = features[:500]    # sample the first epoch from the recording (using only one channel)
X_test = features[500:]
y_train = labels[:500]
y_test = labels[500:]

learning_rate = 0.001  # rate at which to update the parameters
batch_size = 16        # number of samples used before updating parameters
n_epochs = 200            # number of iterations over dataset
batches_per_epoch = len(X_train) // batch_size 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#  def train_loop(input, labels, model, loss_fn, optimizer):
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


# Restore best model
model.load_state_dict(best_weights)
 
# Plot the loss and accuracy
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.show()
 
plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
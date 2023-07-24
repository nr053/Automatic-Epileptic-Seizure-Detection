"""
The second model, taking the raw inputs instead of the features - using only the first channel.
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from p_tools import make_labels, remove_suffix 
from path import Path
import mne
import pandas as pd
import tqdm
import copy
from torch.utils.data import DataLoader
from p_tools import apply_montage

file_name = Path.machine_path + "train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf"
df = pd.read_csv(file_name.removesuffix('edf') + 'csv_bi', header=5)
raw_data = mne.io.read_raw_edf(file_name, infer_types=True)

bipolar_data = apply_montage(raw_data)

epochs = mne.make_fixed_length_epochs(bipolar_data, duration=1) # create epochs
X = torch.tensor(epochs.get_data())
y = make_labels(X, df)

X_train = X[:500]#.permute(0,2,1) #dim: n_epochs, sample length, num_channels
X_test = X[500:]#.permute(0,2,1)
y_train = y[:500]
y_test = y[500:]


torch.set_default_dtype(torch.float64)

class NeuralNetwork(nn.Module):
    """Neural network taking two features (mean,variance) as input and returning class labels"""
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten(0,-1)
        self.linear_stack = nn.Sequential(
            nn.Linear(5120, 1000),
            nn.ReLU(),
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Linear(100,2)
        )
    def forward(self,x):
        #x = self.flatten(x)
        logits = self.linear_stack(x)
        #probs = (softmax(logits))
        #pred = torch.round(probs)
        return logits
    
class PrintSize(nn.Module):
    """Utility to print the size of the tensor in the current step (only on the first forward pass)"""
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x

class NeuralNetwork2(nn.Module):
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
            nn.Flatten(),
            nn.Linear(in_features=3400, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500,out_features=2)
        )
    def forward(self,x):
        logits = self.convolutional_stack(x)
        #probs = softmax(logits)
        #preds = torch.round(probs)
        return logits

#model = NeuralNetwork()
model = NeuralNetwork2()
learning_rate = 1e-3  # rate at which to update the parameters
batch_size = 16        # number of samples used before updating parameters
n_epochs = 200            # number of iterations over dataset
batches_per_epoch = len(X_train) // batch_size
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
            #acc = (torch.argmax(y_pred,1) == torch.argmax(y_batch,1)).float().mean()
            acc = (torch.argmax(y_pred) == torch.argmax(y_batch)).float().mean()
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
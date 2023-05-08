"""
Prove that a CNN can learn to classify a signal epoch. 

Train a CNN on a single channel of a 1 second window and visualise the loss and accuracy as the model trains. 

Being extended to include all 20 channels. This means more network parameters and probably a longer training time.

"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

file_path = "/Users/toucanfirm/Documents/DTU/Speciale/tools/data_small/aaaaaacz/s006_2015_10_05/03_tcp_ar_a/aaaaaacz_s006_t000/100.pt"

X = torch.load(file_path)

assert X.sum().item() != 0.0, ("You are trying to train on an empty tensor!, Please choose another one.")

x = X[0]
x = x[None,:] #add extra dimension to tensor
y = torch.tensor([1.0])
y = y[None,:] #add extra dimension to tensor


class PrintSize(nn.Module):
    """Utility to print the size of the tensor in the current step (only on the first forward pass)"""
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x

class Model(nn.Module):
    """CNN that takes one channel"""
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=2),
            nn.ReLU(),
            PrintSize(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            PrintSize(),
            nn.Conv1d(in_channels=5, out_channels=1, kernel_size=5, padding=2),
            PrintSize(),
            nn.ReLU(),
            nn.Flatten(),
            PrintSize(),
            nn.Linear(in_features=85,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        logits=self.stack(x)
        #print(logits)
        #print(type(logits))
        return logits


class Model_ext(nn.Module):
    """CNN that takes one channel"""
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=30, kernel_size=5, padding=2),
            nn.ReLU(),
            PrintSize(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            PrintSize(),
            nn.Conv1d(in_channels=30, out_channels=10, kernel_size=5, padding=2),
            PrintSize(),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=1, kernel_size=5, padding=2),
            PrintSize(),
            nn.ReLU(),
            nn.Flatten(),
            PrintSize(),
            nn.Linear(in_features=85,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        logits=self.stack(x)
        #print(logits)
        #print(type(logits))
        return logits



print(X)

#model = Model() #model taking only one channel
model = Model_ext()

learning_rate = 1e-4
n_epochs=800
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_hist = []
train_acc_hist = []
train_pred_hist = []
train_label_hist = []

model.train()

for i in tqdm(range(n_epochs)):
    pred = model(x) #using only one channel
    loss = loss_fn(pred,y)

    if pred > 0.75:
        label=1
    else:
        label=0
    
    train_loss_hist.append(loss.item())
    train_acc_hist.append(int(label==y))
    train_pred_hist.append(pred.item())
    train_label_hist.append(label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(train_loss_hist)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_loss_hist.png")
plt.show()

plt.plot(train_pred_hist)
plt.xlabel("epochs")
plt.ylabel("logits")
plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_pred_hist.png")
plt.show()

plt.plot(train_label_hist)
plt.xlabel("epochs")
plt.ylabel("label")
plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_label_hist.png")
plt.show()

plt.plot(train_acc_hist)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_acc_hist.png")
plt.show()
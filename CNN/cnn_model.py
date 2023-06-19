"""
CNN architecture definition.

The inspiration for this model comes from the DTU paper and Applying Deep Learning for Epilepsy Seizure Detection and Brain Mapping Visualization
M. SHAMIM HOSSAIN. 

The input is converted from a 20 channel 1x500 signal, to a single channel 20x500 image. The use of the terms "channel" must be used with care. 

In EEG nomenclature a "channel" refers to the pair of electrodes being compared to produce a time series of voltages.
In CNN architecture a "channel" usually refers to a feature set relating to a specific kernel. 

The first convolutional layer convolves each "channel" in the time direction separately. This aims to extract time series information one one channel at a time. 
The second convolution convolves across the "channels" for each time step. Spatial information is learnt while conserving temporal structure. 


"""



import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
#from sklear.model_selection import StratifiedShuffleSplit
import time 
from progress.bar import Bar
from parent.path import Path


torch.set_default_dtype(torch.float64)
from cnn_dataloader import CNN_Dataset
        
train_data = CNN_Dataset(csv_file= Path.repo + '/TrainingEpochs/train_only_records_with_seizures.csv')
test_data = CNN_Dataset(csv_file= Path.repo + '/DevEpochs/dev.csv')

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)




class PrintSize(nn.Module):
    """Utility to print the size of the tensor in the current step (only on the first forward pass)"""
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten(0, -1)
        self.convolutional_stack = nn.Sequential(
            #nn.LayerNorm(normalized_shape = [256]),
            PrintSize(),
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1,10), padding=0, stride=1), #convolve in the time direction with (1x10) filters. 

            PrintSize(),

            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(20,1), padding=(10,0)),
            PrintSize(),
            nn.BatchNorm2d(num_features=20),
            PrintSize(),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),

            PrintSize(),
            
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(20,10), padding=0, stride=1),
            PrintSize(),
            nn.BatchNorm2d(num_features=40),
            PrintSize(),
            nn.ELU(),            
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            
            PrintSize(),

            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(40,10), padding=0, stride=1),
            PrintSize(),
            nn.BatchNorm2d(num_features=80),
            PrintSize(),
            nn.ELU(),

            nn.Flatten(1,-1), #keep the first dimension (batch size) and flatten the rest
            PrintSize(),
            nn.Linear(in_features=500, out_features=1),
            nn.softmax()

    def forward(self,x):
        logits = self.convolutional_stack(x)
        #probs = softmax(logits)
        #preds = torch.round(probs)
        return logits

# class CNN2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #self.flatten = nn.Flatten(0, -1)
#         self.convolutional_stack = nn.Sequential(
#             #nn.LayerNorm(normalized_shape = [256]),
#             PrintSize(),
#             nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3,10), padding=(1,4), stride=1), #convolve across time and channels with (3x10) filters. 
#             nn.ReLU(),
#             PrintSize(),
#             nn.MaxPool1d(kernel_size=3, stride=3),
#             PrintSize(),
#             nn.Conv1d(in_channels=40,out_channels=80, kernel_size=5, padding=2),
#             PrintSize(),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=80,out_channels=40, kernel_size=5, padding=2),
#             PrintSize(),
#             nn.ReLU(),
#             nn.Flatten(1,-1), #keep the first dimension (batch size) and flatten the rest
#             PrintSize(),
#             nn.Linear(in_features=3400, out_features=500),
#             PrintSize(),
#             nn.ReLU(),
#             nn.Linear(in_features=500,out_features=1),
#             PrintSize(),
#             nn.Sigmoid()
#         )
#     def forward(self,x):
#         logits = self.convolutional_stack(x)
#         #probs = softmax(logits)
#         #preds = torch.round(probs)
#         return logits


model = CNN1()


learning_rate = 4e-6  # rate at which to update the parameters
n_epochs = 1            # number of iterations over dataset
batch_size = 512
#loss_fn = nn.BCELoss()
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

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


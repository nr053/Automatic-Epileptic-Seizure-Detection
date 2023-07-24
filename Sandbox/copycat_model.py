"""
Copy cat model of the P-1D-CNN structure outlined in 

'An Automated System for Epilepsy Detection using EEG Brain Signals based on Deep Learning Approach'

"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne
from p_tools import load_data
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)



#Load the original EDF file and plot the signal along with it's annotations to get a sense of what we are training/testing on. 

#edf_path = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaacyf/s009_2015_04_01/01_tcp_ar/aaaaacyf_s009_t001.edf"
#edf_path_2 = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaacyf/s009_2015_04_01/01_tcp_ar/aaaaacyf_s009_t000.edf"

#edf_path = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaaaac/s001_2002_12_23/02_tcp_le/aaaaaaac_s001_t000.edf"
#edf_path_2 = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaaaac/s001_2002_12_23/02_tcp_le/aaaaaaac_s001_t001.edf"




#data,epoch_tensor,labels = load_data(edf_path, epoch_length=1)
#data2, epoch_tensor2, labels2 = load_data(edf_path_2, epoch_length=1)

#data.plot(duration=5,highpass=1, lowpass=70)


#file_path = "/Users/toucanfirm/Documents/DTU/Speciale/tools/data_small/aaaaaacz/s006_2015_10_05/03_tcp_ar_a/aaaaaacz_s006_t000/100.pt"
#file_path = "/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_small/aaaaacyf/s009_2015_04_01/01_tcp_ar/aaaaacyf_s009_t001/0.pt"


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
        one_channel_tens = epoch_tens[0]
        one_channel_tens = one_channel_tens[None, :]

        label = torch.tensor([1 - float(self.df.iloc[idx,2]), float(self.df.iloc[idx,2])])

        sample = {'X': one_channel_tens, 'y': label}

        return sample



batch_size=1
dataset = EEG_Dataset(csv_file="epoch_data_small.csv")

train_size = int(len(dataset)*0.7)
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size,test_size])

#only use a small fraction of the dataset to test if the model works at all
train_subset = Subset(train_data, [0,1,2,3,4])
test_subset = Subset(test_data, [5,6,7])

train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)


#X = torch.load(file_path) #load tensor from .pt file
#X = epoch_tensor[0:10]
#X_test = epoch_tensor[10]
#assert X.sum().item() != 0.0, ("You are trying to train on an empty tensor!, Please choose another one.")

#x = X[0] #just one channel
#x = x[None,:] #add extra dimension to tensor
#y = torch.ones([10,1])
#y_test = torch.tensor([[1.0]])
#y = y[None,:] #add extra dimension to tensor


class PrintSize(nn.Module):
    """Utility to print the size of the tensor in the current step (only on the first forward pass)"""
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x


class CopyCat(nn.Module):
    """P-1D-CNN copy cat model
    
    - Pyramid architecture
    - No pooling layer
    - Bigger strides in convolutional layers
    - Softmax classifier in last layer"""

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=5, stride=3),
            nn.BatchNorm1d(num_features=24),
            nn.ReLU(),
            
            PrintSize(),

            nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),

            PrintSize(),
            
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),

            PrintSize(),

            nn.Flatten(),
            
            PrintSize(),

            nn.Linear(in_features=160, out_features=20),
            nn.ReLU(),

            #Drouput()

            nn.Linear(in_features=20, out_features=2),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        probs = self.stack(x)
        return probs

model = CopyCat() #model taking only one channel

learning_rate = 1e-3
n_epochs=100
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_hist = []
train_acc_hist = []
train_pred_hist = []
#train_label_hist = []
test_acc_hist = []
test_loss_hist = []

first_pass = True # a dummy variable to print values on the first pass of the network

#train_loss_is_large = True
#test_loss_is_large = True

model.train()

for i in tqdm(range(n_epochs)):
    loop = tqdm(train_dataloader)
    for batch, sample in enumerate(loop):
        if first_pass:
            print(batch, sample)
            print(sample['X'].shape)
            first_pass = False
        
        pred = model(sample['X']) #using all 20 channels
        loss = loss_fn(pred,sample['y'])

        #if loss.item() < 0.05 and train_loss_is_large: #print iteration number when loss is sufficiently low
        #    print(f"Zero loss at step: {i}")
        #    train_loss_is_large = False

        #if pred > 0.75:
        #    label=1
        #else:
        #    label=0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss_hist.append(loss.item())
    train_acc_hist.append((sample['y']==pred.round()).float().mean().item())
    train_pred_hist.append(pred.mean().item())
    #train_label_hist.append(label)




        #test accuracy at each step
        #model.eval() #put model in evaluation mode
        #pred_test = model(X_test)
        #loss_test = loss_fn(pred_test, y_test)

        #if loss_test.item() < 0.05 and test_loss_is_large: #print iteration number when loss is sufficiently low
        #    print(f"Zero test loss at step: {i}")
        #    test_loss_is_large = False

        #test_loss_hist.append(loss_test.item())
        #test_pred_hist.append(pred_test.item())

model.eval()

#loop = tqdm(test_dataloader)
with torch.no_grad(): #disable gradient calculation to save memory consumption
    for sample in tqdm(test_dataloader):
        pred = model(sample['X'])
        loss = loss_fn(pred, sample['y'])

        test_loss_hist.append(loss.item())
        test_acc_hist.append((sample['y']==pred.round()).float().mean().item())


plt.figure(0)
plt.plot(train_loss_hist, label="train")
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
#plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_loss_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/copycat_train_loss_hist.png")
#plt.show()

plt.figure(1)
plt.plot(train_pred_hist, label="train")
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
#plt.plot(test_pred_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("logits")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_pred_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/copycat_train_pred_hist.png")
#plt.show()

# plt.figure(2)
# plt.plot(train_label_hist)
# plt.xlabel("epochs")
# plt.ylabel("label")
# plt.legend()
# #plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_label_hist.png")
# plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/train_label_hist.png")
# #plt.show()

plt.figure(3)
plt.plot(train_acc_hist)
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_acc_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/copycat_train_acc_hist.png")
#plt.show()


plt.figure(4)
plt.plot(test_loss_hist, label="test")
plt.xlabel("batch")
plt.ylabel("loss")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_loss_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/copycat_test_loss_hist.png")
#plt.show()

plt.figure(5)
plt.plot(test_acc_hist, label="test")
plt.xlabel("batch")
plt.ylabel("accuracy")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_pred_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/copycat_test_acc_hist.png")
#plt.show()

print(model)
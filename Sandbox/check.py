"""
Prove that a CNN can learn to classify a signal epoch. 

Train a CNN on a single channel of a 1 second window and visualise the loss and accuracy as the model trains. 

Being extended to include all 20 channels. This means more network parameters and probably a longer training time.

Extended to train on the first 10 seconds and tested on the 11th. 

Extended even further to train on 40 epochs of background and 40 epochs of seizure 

"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne
from p_tools import load_data
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)



#Load the original EDF file and plot the signal along with it's annotations to get a sense of what we are training/testing on. 

edf_path = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaacyf/s009_2015_04_01/01_tcp_ar/aaaaacyf_s009_t001.edf"
data,epoch_tensor,labels = load_data(edf_path, epoch_length=1)
#data.plot(duration=5,highpass=1, lowpass=70)


#file_path = "/Users/toucanfirm/Documents/DTU/Speciale/tools/data_small/aaaaaacz/s006_2015_10_05/03_tcp_ar_a/aaaaaacz_s006_t000/100.pt"
file_path = "/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_small/aaaaacyf/s009_2015_04_01/01_tcp_ar/aaaaacyf_s009_t001/0.pt"


class OneEDF_Dataset(Dataset):
    """A dataset containing a single recording"""
    def __init__(self, epoch_tensor, labels, transform=None):
        """Takes a tensor of time windows that cover the entire recording
        along with their class labels"""
        self.epoch_tensor = epoch_tensor[330:410] #use select range of epochs to balance classes
        self.labels = labels[330:410,[1]] #only take the single binary class value. 'Bckg'=0, 'Seiz'=1 - also within select range
        self.transform = transform

    def __len__(self):
        return len(self.epoch_tensor)

    def __getitem__(self, idx):
        sample = {'X': self.epoch_tensor[idx], 'y': self.labels[idx]}
        return sample

dataset = OneEDF_Dataset(epoch_tensor, labels)
dataloader = DataLoader(dataset, batch_size=40, shuffle=True)

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
    """CNN that takes all 20 channels"""
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=50, kernel_size=9, padding=4),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU(),
            PrintSize(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=9, padding=4),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU(),
            PrintSize(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            PrintSize(),
            nn.Conv1d(in_channels=50, out_channels=10, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=10),
            PrintSize(),
            nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=1, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_features=1),
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


# class CopyCat(nn.Module):
#     """P-1D-CNN copy cat model
    
#     - Pyramid architecture
#     - No pooling layer
#     - Bigger strides in convolutional layers
#     - Softmax classifier in last layer"""

#     def __init__(self):
#         super().__init__()
#         self.stack = nn.Sequential(
#             nn.Conv1d()
#             nn.BatchNorm()
#             nn.ReLU()

#             nn.Conv1d()
#             nn.BatchNorm()
#             nn.ReLU()

#             nn.Conv1d()
#             nn.BatchNorm()
#             nn.ReLU()

#             nn.FC(),
#             nn.ReLU()

#             Drouput()

#             nn.FC()

#             nn.softmax()
#         )

#print(X)

#model = Model() #model taking only one channel
model = Model_ext()

learning_rate = 1e-3
n_epochs=400
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_hist = []
train_acc_hist = []
train_pred_hist = []
#train_label_hist = []
#test_pred_hist = []
#test_loss_hist = []

first_pass = True # a dummy variable to print values on the first pass of the network

train_loss_is_large = True
#test_loss_is_large = True

for i in tqdm(range(n_epochs)):
    for batch, sample in enumerate(dataloader):
        model.train()
        #pred = model(x) #using only one channel
        if first_pass:
            print(batch, sample)
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


plt.figure(0)
plt.plot(train_loss_hist, label="train")
#plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_loss_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/train_loss_hist.png")
#plt.show()

plt.figure(1)
plt.plot(train_pred_hist, label="train")
#plt.plot(test_pred_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("logits")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_pred_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/train_pred_hist.png")
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
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_acc_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/train_acc_hist.png")
#plt.show()


# plt.figure(4)
# plt.plot(test_loss_hist, label="test")
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.legend()
# #plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_loss_hist.png")
# plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/test_loss_hist.png")
# #plt.show()

# plt.figure(5)
# plt.plot(test_pred_hist, label="test")
# plt.xlabel("epochs")
# plt.ylabel("logits")
# plt.legend()
# #plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_pred_hist.png")
# plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/test_pred_hist.png")
# #plt.show()
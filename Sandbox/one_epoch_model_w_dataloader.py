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
from p_tools import load_data, write_annotations
from annotate_predictions import prediction_to_annotation
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from statistics import mean
import time
import numpy as np
import copy

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
        #self.df = pd.read_csv(csv_file)[0:500]
        self.df['pred'] = np.nan
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        epoch_path = self.df.iloc[idx,1]
        epoch_tens = torch.load(epoch_path)

        #label = torch.tensor([1 - float(self.df.iloc[idx,2]),float(self.df.iloc[idx,2])]) #k-hot multiclass format
        label = torch.tensor([float(self.df.iloc[idx,2])]) #singular logit output

        sample = {'idx': idx, 'X': epoch_tens, 'y': label}

        return sample

    def addprediction(self, idx, predictions):
        self.df.loc[idx, ['pred']] = predictions.detach().numpy()


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
            nn.Linear(in_features=10, out_features=2),
            nn.Sigmoid()
            #nn.Softmax(dim=0)  #softmax because we want the probabilities to add to one. 
        )



    def forward(self,x):
        logits=self.stack(x)
        #print(logits)
        #print(type(logits))
        return logits


class Model_ext_256(nn.Module):
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
            #nn.Linear(in_features=10, out_features=2),     #k-hot encoded output
            nn.Linear(in_features=10, out_features=1),      #singular probability output
            nn.Sigmoid()
            #nn.Softmax(dim=0)                              #softmax because we want the probabilities to add to one. 
        )

    def forward(self,x):
        logits=self.stack(x)
        #print(logits)
        #print(type(logits))
        return logits


# class Model_ext_250(nn.Module):
#     """CNN that takes all 20 channels"""
#     def __init__(self):
#         super().__init__()
#         self.stack = nn.Sequential(
#             nn.Conv1d(in_channels=20, out_channels=50, kernel_size=9, padding=4),
#             nn.BatchNorm1d(num_features=50),
#             nn.ReLU(),
#             PrintSize(),
#             nn.Conv1d(in_channels=50, out_channels=50, kernel_size=9, padding=4),
#             nn.BatchNorm1d(num_features=50),
#             nn.ReLU(),
#             PrintSize(),
#             nn.MaxPool1d(kernel_size=3, stride=3),
#             PrintSize(),
#             nn.Conv1d(in_channels=50, out_channels=10, kernel_size=7, padding=3),
#             nn.BatchNorm1d(num_features=10),
#             PrintSize(),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=10, out_channels=1, kernel_size=7, padding=3),
#             nn.BatchNorm1d(num_features=1),
#             PrintSize(),
#             nn.ReLU(),
#             nn.Flatten(),
#             PrintSize(),
#             nn.Linear(in_features=83,out_features=10),
#             nn.ReLU(),
#             nn.Linear(in_features=10, out_features=2),
#             nn.Sigmoid()
#         )

#     def forward(self,x):
#         logits=self.stack(x)
#         #print(logits)
#         #print(type(logits))
#         return logits


# class CopyCat(nn.Module):
#     """P-1D-CNN copy cat model
    
#     - Pyramid architecture
#     - No pooling layer
#     - Bigger strides in convolutional layers
#     - Softmax classifier in last layer"""

#     def __init__(self):
#         super().__init__()
#         self.stack = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=24, kernel_size=5, stride=3),
#             nn.BatchNorm1d(num_features=24),
#             nn.ReLU(),

#             nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3, stride=2),
#             nn.BatchNorm1d(num_features=16),
#             nn.ReLU(),

#             nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
#             nn.BatchNorm()
#             nn.ReLU()

#             nn.FC(),
#             nn.ReLU()

#             Drouput()

#             nn.FC()

#             nn.softmax()
#         )



dataset = EEG_Dataset(csv_file="balanced_data_small.csv")
train_size = int(len(dataset)*0.7)
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size,test_size])

# Print information about the train and test data set. 
no_train_seizures = 0
no_test_seizures = 0

print("------------------")
print("Analysing train data")
print("------------------")

for i in tqdm(range(len(train_data))):
    no_train_seizures += train_data[i]['y']

print("------------------")
print("Analysing test data")
print("------------------")

for i in tqdm(range(len(test_data))):    
    no_test_seizures += test_data[i]['y']


print("-------------")
print(f"Number of seizure epochs in training set: {no_train_seizures.item()}")
print(f"Ratio of seizures to background noise: {no_train_seizures.item() / len(train_data)}")
print(f"Number of seizure epochs in test set: {no_test_seizures.item()}")
print(f"Ratio of seizures to background noise: {no_test_seizures.item() / len(test_data)}")
print("-------------")


batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True) #batch size = 64. Give the model 64 samples at a time 

#X = torch.load(file_path) #load tensor from .pt file
#X = epoch_tensor[0:10]
#X_test = epoch_tensor[10]
#assert X.sum().item() != 0.0, ("You are trying to train on an empty tensor!, Please choose another one.")

#x = X[0] #just one channel
#x = x[None,:] #add extra dimension to tensor
#y = torch.ones([10,1])
#y_test = torch.tensor([[1.0]])
#y = y[None,:] #add extra dimension to tensor


#print(X)

#model = Model() #model taking only one channel
model = Model_ext_256()
#model = Model_ext_250()

learning_rate = 5e-4
n_epochs=300
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_epoch_loss_history = []
train_epoch_acc_history = []
test_epoch_loss_history = []
test_epoch_acc_history = []

best_accuracy = -np.inf
best_weights = None

first_pass = True # a dummy variable to print values on the first pass of the network

#train_loss_is_large = True
#test_loss_is_large = True



for i in tqdm(range(n_epochs)):

    train_batch_loss_history = []                      #array to hold the average train loss over each batch 
    train_batch_acc_history = []                       #array to hold the average train accuracy over each batch 
    test_batch_loss_history = []                      #array to hold the average test loss over each batch 
    test_batch_acc_history = []                       #array to hold the average test accuracy over each batch 

    model.train()

    train_loop = tqdm(train_dataloader)
    for batch, sample in enumerate(train_loop):
        #pred = model(x) #using only one channel
        # if first_pass:
        #     print(batch, sample)
        #     first_pass = False
        
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
        
        train_batch_loss_history.append(loss.mean().item())
        train_batch_acc_history.append((sample['y']==pred.round()).float().mean().item())

        
        #train_pred_hist.append(pred.mean().item())
        #train_label_hist.append(label)
    
    
    train_epoch_loss_history.append(mean(train_batch_loss_history))       #average loss over the epoch stored in history
    train_epoch_acc_history.append(mean(train_batch_acc_history))    #average accuracy over the epoch stored in history


     
    #test the model after each iteration through the whole training set (epoch). 
    #calculate the loss and accuracy across the whole test set 
    #save the sensitivity and specificity 

    model.eval()                                                             #put model in evaluation mode
    

#    prediction_history = []
#    label_history = []

    loop = tqdm(test_dataloader)
    for test_sample in loop:                                  #load the only batch which is every sample in the full test set
        pred_test = model(test_sample['X'])                                      #make prediction with the model
        loss_test = loss_fn(pred_test, test_sample['y'])                #calculate the loss for the predictions

        test_batch_loss_history.append(loss_test.mean().item())
        test_batch_acc_history.append((test_sample['y']==pred_test.round()).float().mean().item())

#        prediction_history.extend(pred_test.round().detach().squeeze().tolist())
#        label_history.extend(test_sample['y'].detach().squeeze().tolist())

    test_epoch_loss_history.append(mean(test_batch_loss_history))       #average loss over the epoch stored in history
    test_epoch_acc_history.append(mean(test_batch_acc_history))


    if i % 10 == 0:
        print("-----------")
        print(f"At epoch number {i}")
        print(f"Training loss: {train_epoch_loss_history[-1]}")
        print(f"Test loss: {test_epoch_loss_history[-1]}")
        print(f"Training accuracy: {train_epoch_acc_history[-1]}")
        print(f"Test accuracy: {test_epoch_acc_history[-1]}")
        print("-----------")

    if test_epoch_acc_history[-1] > best_accuracy:  #save the best performing model weights
        best_accuracy = test_epoch_acc_history[-1]
        best_model_state = copy.deepcopy(model.state_dict())

    #sklearns confusion matrix


print("----------------")
print("Finished training! Reloading the best model and running final loop to calculate performance metrics.")
print("----------------")

#reload best model
#model = Model_ext_256()
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), "/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/models/CNN_model.pt")
model.eval()

final_accuracies = []
final_predictions = []
final_labels = []

#test the model on the full test set 
loop = tqdm(test_dataloader)
for final_sample in loop:
    pred_final = model(final_sample['X'])
    dataset.addprediction(final_sample['idx'].tolist(), pred_final.round()) #add predictions to original dataframe
    
    final_accuracies.extend((final_sample['y'] == pred_final.round()).float().squeeze().tolist())
    final_predictions.extend(pred_final.round().detach().squeeze().tolist())
    final_labels.extend(final_sample['y'].squeeze().tolist())


model_accuracy = mean(final_accuracies)
print("----------------")
print(f"Final model accuracy: {model_accuracy}")



CM = 0
CM += confusion_matrix(final_labels, final_predictions, labels=[0,1])
tn = CM[0][0]
tp = CM[1][1]
fp = CM[0][1]
fn = CM[1][0]

if ((tp+fn) == 0) or ((tn+fp) == 0):
    print("-------------")
    print("Could not calculate specificity/sensitivity. Zero Division Error. Consider balancing the dataset")
    print("-------------")
else:
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

#my own method of calculating the tn/tp/fn/fp

confusion_vector = torch.tensor(final_predictions) / torch.tensor(final_labels)

true_positives = torch.sum(confusion_vector == 1).item()
false_positives = torch.sum(confusion_vector == float('inf')).item()
true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
false_negatives = torch.sum(confusion_vector == 0).item()


print("Here is the confusion matrix calculated by 'SKlearn")
print(CM)
print(f"True positives: {tp}")
print(f"False positives: {fp}")
print(f"True negatives: {tn}")
print(f"False negatives: {fn}")
print("------------------------")
print("Here is the confusion matrix as calculated by yours truly")
print(f"True positives: {true_positives}")
print(f"False positives: {false_positives}")
print(f"True negatives: {true_negatives}")
print(f"False negatives: {false_negatives}")


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = CM, display_labels = [0, 1])

plt.figure(6)
cm_display.plot()
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/CNN_baseline/confusion.png")

    # # # # prediction_history = []
    # # # # label_history = []

    # # # # one_test_start_time = time.time()

    # # # # loop = tqdm(test_dataloader_one)
    # # # # for batch, sample in enumerate(loop):
    # # # #     pred_test_one = model(sample['X'])
        
    # # # #     prediction_history.append(pred_test_one.round().item())
    # # # #     label_history.append(sample['y'].item())
        
    # # # #     loss_test_one = loss_fn(pred_test_one, sample['y'])
    # # # #     accuracy_test_one = (pred_test_one.round() == sample['y']).float()

    # # # #     test_loss_history_one.append(loss_test_one.item())
    # # # #     test_acc_history_one.append(accuracy_test_one.item())

    # # # # test_loss_history = mean(test_loss_history_one)
    # # # # test_accuracy_history = mean(test_acc_history_one)

    # # # # one_test_end_time = time.time()

    # # # # #sklearns confusion matrix

    # # # # CM = 0
    # # # # CM += confusion_matrix(label_history, prediction_history, labels=[0,1])
    # # # # tn = CM[0][0]
    # # # # tp = CM[1][1]
    # # # # fp = CM[0][1]
    # # # # fn = CM[1][0]
    
    # # # # if (tp+fn) == 0 or (tn+fp) == 0:
    # # # #     print("-------------")
    # # # #     print("Could not calculate specificity/sensitivity. Zero Division Error. Consider balancing the dataset")
    # # # #     print("-------------")
    # # # # else:
    # # # #     sensitivity = tp / (tp + fn)
    # # # #     specificity = tn / (tn + fp)

    # # # # #my own method of calculating the tn/tp/fn/fp

    # # # # confusion_vector = torch.tensor(prediction_history) / torch.tensor(label_history)

    # # # # true_positives = torch.sum(confusion_vector == 1).item()
    # # # # false_positives = torch.sum(confusion_vector == float('inf')).item()
    # # # # true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    # # # # false_negatives = torch.sum(confusion_vector == 0).item()

    
    # # # # print("Here is the confusion matrix calculated by 'SKlearn")
    # # # # print(CM)
    # # # # print(f"True positives: {tp}")
    # # # # print(f"False positives: {fp}")
    # # # # print(f"True negatives: {tn}")
    # # # # print(f"False negatives: {fn}")
    # # # # print("------------------------")
    # # # # print("Here is the confusion matrix as calculated by yours truly")
    # # # # print(f"True positives: {true_positives}")
    # # # # print(f"False positives: {false_positives}")
    # # # # print(f"True negatives: {true_negatives}")
    # # # # print(f"False negatives: {false_negatives}")

# # # # print("----------------------")
# # # # print("The full dataset as one batch took: ", full_test_end_time - full_test_start_time)
# # # # print("Giving one batch at a time took:", one_test_end_time - one_test_start_time)
# # # # print("----------------------")
# model.eval()

# #loop = tqdm(test_dataloader)
# with torch.no_grad(): #disable gradient calculation to save memory consumption
#     for sample in tqdm(test_dataloader):
#         pred = model(sample['X'])
#         loss = loss_fn(pred, sample['y'])

#         test_loss_hist.append(loss.item())
#         test_acc_hist.append((sample['y']==pred.round()).float().mean().item())



# Sensitivity is the true positive rate. Out of all the positive samples, how many were classified correctly?
# TP / TP + FN





#Specificity is the true negative rate. Models ability to identify negative classes. 
# TN / TN + FP



dataset.df.to_csv("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/results/CNN_preds.csv")
#signals, df = prediction_to_annotation(dataset.df)
# raw = mne.io.read_raw_edf("/home/migo/TUHP/" + signals[0][0], infer_types=True)
# annotations = mne.Annotations(signals[0][1], duration=1.0, description="seizure")
# raw.set_annotations(annotations)
# raw.plot()




plt.figure(0)
plt.plot(train_epoch_loss_history, label="train")
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
#plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_loss_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/CNN_baseline/train_loss_hist.png")
#plt.show()

plt.figure(1)
plt.plot(train_epoch_acc_history, label="train")
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
#plt.plot(test_pred_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_pred_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/CNN_baseline/train_accuracy_hist.png")
#plt.show()



plt.figure(3)
plt.plot(test_epoch_loss_history, label="test")
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
#plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_loss_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/CNN_baseline/test_loss_hist.png")
#plt.show()

plt.figure(4)
plt.plot(test_epoch_acc_history, label="test")
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
#plt.plot(test_pred_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
#plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_pred_hist.png")
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/CNN_baseline/test_accuracy_hist.png")
#plt.show()


# plt.figure(2)
# plt.plot(train_label_hist)
# plt.xlabel("epochs")
# plt.ylabel("label")
# plt.legend()
# #plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_label_hist.png")
# plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/train_label_hist.png")
# #plt.show()



# plt.figure(2)
# plt.plot(test_loss_hist, label="test")
# plt.xlabel("batch")
# plt.ylabel("loss")
# plt.legend()
# #plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_loss_hist.png")
# plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/test_loss_hist.png")
# #plt.show()

# plt.figure(3)
# plt.plot(test_acc_hist, label="test")
# plt.xlabel("batch")
# plt.ylabel("accuracy")
# plt.legend()
# #plt.savefig("/Users/toucanfirm/Documents/DTU/Speciale/One_epoch_model/train_pred_hist.png")
# plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/plots/test_acc_hist.png")
# #plt.show()

print(model)
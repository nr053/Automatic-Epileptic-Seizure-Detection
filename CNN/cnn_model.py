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
from sklearn.metrics import confusion_matrix
from statistics import mean
import copy
from sklearn import metrics

torch.set_default_dtype(torch.float64)
from cnn_dataloader import CNN_Dataset

st = time.time()
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

train_data = CNN_Dataset(csv_file= Path.repo + '/TrainingEpochs/train_only_records_with_seizures.csv')
test_data = CNN_Dataset(csv_file= Path.repo + '/DevEpochs/dev_only_records_with_seizures.csv')



class PrintSize(nn.Module):
    """Utility to print the size of the tensor in the current step (only on the first forward pass)"""
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x

# class CNN1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #self.flatten = nn.Flatten(0, -1)
#         self.convolutional_stack = nn.Sequential(
#             #nn.LayerNorm(normalized_shape = [256]),
#             PrintSize(),
#             nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1,10), padding=0, stride=1), #convolve in the time direction with (1x10) filters. 

#             PrintSize(),

#             nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(20,1), padding=(10,0)),
#             PrintSize(),
#             nn.BatchNorm2d(num_features=20),
#             PrintSize(),
#             nn.ELU(),
#             nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),

#             PrintSize(),
            
#             nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(10,10), padding=0, stride=1),
#             PrintSize(),
#             nn.BatchNorm2d(num_features=40),
#             PrintSize(),
#             nn.ELU(),            
#             nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            
#             PrintSize(),

#             nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(10,10), padding=0, stride=1),
#             PrintSize(),
#             nn.BatchNorm2d(num_features=80),
#             PrintSize(),
#             nn.ELU(),

#             nn.Flatten(1,-1), #keep the first dimension (batch size) and flatten the rest
#             PrintSize(),
#             nn.Linear(in_features=26160, out_features=2),
#             nn.Softmax(dim=1)
#         )

#     def forward(self,x):
#         logits = self.convolutional_stack(x)
#         #probs = softmax(logits)
#         #preds = torch.round(probs)
#         return logits

# class CNN2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #self.flatten = nn.Flatten(0, -1)
#         self.convolutional_stack = nn.Sequential(
#             #nn.LayerNorm(normalized_shape = [256]),
#             PrintSize(),

#             nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3,10), padding=(1,20), stride=(1,3)),
#             PrintSize(),
#             nn.BatchNorm2d(num_features=5),
#             PrintSize(),
#             nn.ELU(),
#             nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),

#             PrintSize(),
            
#             nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(10,10), padding=(10,4), stride=(4,2)),
#             PrintSize(),
#             nn.BatchNorm2d(num_features=10),
#             PrintSize(),
#             nn.ELU(),            
#             nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            
#             PrintSize(),

#             #nn.Dropout(p=0.5),

#             nn.Flatten(1,-1), #keep the first dimension (batch size) and flatten the rest
#             PrintSize(),
#             nn.Linear(in_features=1760, out_features=2),
#             PrintSize(),
#             nn.Softmax(dim=1)
#         )

#     def forward(self,x):
#         logits = self.convolutional_stack(x)
#         #probs = softmax(logits)
#         #preds = torch.round(probs)
#         return logits


class CNN3(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten(0, -1)
        self.convolutional_stack = nn.Sequential(
            #nn.LayerNorm(normalized_shape = [256]),
            PrintSize(),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=(1,1), stride=(3,5)),
            PrintSize(),
            nn.BatchNorm2d(num_features=1),
            PrintSize(),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),

            PrintSize(),

            #nn.Dropout(p=0.5),

            nn.Flatten(1,-1), #keep the first dimension (batch size) and flatten the rest
            PrintSize(),
            nn.Linear(in_features=350, out_features=2),
            PrintSize(),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        logits = self.convolutional_stack(x)
        #probs = softmax(logits)
        #preds = torch.round(probs)
        return logits

#model = CNN1()
#model = CNN2()
model = CNN3()

learning_rate = 4e-6  # rate at which to update the parameters
n_epochs = 1            # number of iterations over dataset
batch_size = 512
#batches_per_epoch = len(train_data) / batch_size
#loss_fn = nn.BCELoss()
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# best_acc = - np.inf
# best_weights = None
# train_loss_hist = []
# train_acc_hist = []



#model.train()  # set the model to training mode (good practice)


# for epoch in range(n_epochs):
#     #print("Training step")
#     model.train()
#     epoch_loss = []
#     epoch_acc = []
#     for batch, sample in tqdm(enumerate(train_dataloader)):
#         #loop_start = time.time()
#         # forward pass
#         x_input = sample['X']
#         y_gt = sample['y']
#         #pred_start = time.time()
#         y_pred = model(x_input)
#         #pred_stop = time.time()
#         loss = loss_fn(y_pred, y_gt)
#         #checkpoint1 = time.time()
#         #print(f"Prediction and loss took: {checkpoint1 - loop_start}")
#         #print(f"Prediction took: {pred_stop - pred_start}")
#         # backward pass
#         #print("optimisation step")
#         optimizer.zero_grad()
#         loss.backward()
#         # update weights
#         optimizer.step()
#         # compute and store epoch metrics
#         #checkpoint2 = time.time()
#         #print(f"Optimisation step: {checkpoint2 - checkpoint1}")
#         #print("calculating metrics")
#         batch_accuracy = (torch.argmax(y_pred,1) == torch.argmax(y_gt,1)).float().mean()
#         epoch_loss.append(float(loss))
#         epoch_acc.append(float(batch_accuracy))
#         #checkpoint3 = time.time()
#         #print(f"Metrics took: {checkpoint3 - checkpoint2}")
#         #bar.set_postfix(loss=float(loss),accuracy=float(acc))

#     #print("Eval step")
#     model.eval() #put the model in evaluation mode, vital when network contains dropout/normalisation layers
#     for batch, sample in tqdm(enumerate(test_dataloader)):
#         y_pred = model(sample['X'])
#         loss_test = float(loss_fn(y_pred, sample['y']))
#         accuracy = float((torch.argmax(y_pred,1) == torch.argmax(sample['y'],1)).float().mean())
#         train_loss_hist.append(np.mean(epoch_loss))
#         train_acc_hist.append(np.mean(epoch_acc))
#         test_loss_hist.append(loss_test)
#         test_acc_hist.append(accuracy)
#         # if accuracy > best_acc:
#         #     best_acc = acc
#         #     best_weights = copy.deepcopy(model.state_dict())
#         print(f"Epoch {epoch} validation: loss={loss}, accuracy={accuracy}")

def train_loop(dataloader, model, loss_fn, optimizer):
    train_batch_loss_history = []
    train_batch_accuracy_history = []
    size = len(dataloader.dataset)

    pred_times = []
    optimisation_times = []


    #for batch, sample in tqdm(enumerate(dataloader)):
    #with Bar('Processing...') as bar:
    loop = tqdm(dataloader)
    for sample in loop:
    #for sample in dataloader:
        t_before_pred = time.time()
        sample['prediction'] = model(sample['X'])
        t_after_pred = time.time()
        loss = loss_fn(sample['prediction'], sample['y'])

        train_batch_loss_history.append(loss.item())
        batch_accuracy = (torch.argmax(sample['prediction'],1) == torch.argmax(sample['y'],1)).float().mean().item()
        train_batch_accuracy_history.append(batch_accuracy)

        t_before_optimisation = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_after_optimisation = time.time()

        #    bar.next()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch+1)*len(sample['X'])
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #     #print(f"Pred = {pred}")

        pred_times.append(t_after_pred - t_before_pred)
        optimisation_times.append(t_after_optimisation - t_before_optimisation )

    return train_batch_loss_history, train_batch_accuracy_history, mean(pred_times), mean(optimisation_times)

def test_loop(dataloader, model, loss_fn):
    test_batch_loss_history = []
    test_batch_accuracy_history = []
    test_append_times = []

    with torch.no_grad():
        loop = tqdm(dataloader)
        for sample in loop:
        #for sample in dataloader:
            #make prediction using model
            sample['prediction'] = model(sample['X'])
            
            t_array_start = time.time()
            #calculate batch loss and accuracy, store in history arrays 
            test_batch_loss_history.append(loss_fn(sample['prediction'], sample['y']).item())
            test_batch_accuracy_history.append((torch.argmax(sample['prediction'],1) == torch.argmax(sample['y'],1)).float().mean().item())
            t_array_end = time.time()

            test_append_times.append(t_array_end - t_array_start)

    return test_batch_loss_history, test_batch_accuracy_history, mean(test_append_times) 


train_epoch_loss_history = []
train_epoch_accuracy_history = []
test_epoch_loss_history = []
test_epoch_accuracy_history = []

best_accuracy = 0

opt_times = []
predd_times = []
train_times = []
test_times = []

#iterate over the entire dataset n=n_epochs times
for epoch in tqdm(range(n_epochs)):
    train_start = time.time()
    model.train()
    train_batch_loss_history, train_batch_accuracy_history, pred_time, optimisation_time = train_loop(train_dataloader, model, loss_fn, optimizer)
    train_end = time.time()

    test_start = time.time()
    model.eval()
    test_batch_loss_history, test_batch_accuracy_history, test_append_time = test_loop(test_dataloader, model, loss_fn)
    test_end = time.time()

    train_epoch_loss_history.append(mean(train_batch_loss_history))
    train_epoch_accuracy_history.append(mean(train_batch_accuracy_history))
    test_epoch_loss_history.append(mean(test_batch_loss_history))
    test_epoch_accuracy_history.append(mean(test_batch_accuracy_history))

    opt_times.append(optimisation_time)
    predd_times.append(pred_time)
    test_times.append(test_append_time)

    if test_epoch_accuracy_history[-1] > best_accuracy:
        best_accuracy = test_epoch_accuracy_history[-1]
        best_weights = copy.deepcopy(model.state_dict())

# Restore best model
model.load_state_dict(best_weights)
torch.save(model.state_dict(), Path.repo + '/CNN/models/CNN_model_weights.pt')
model.eval()

final_batch_accuracy_history = []
final_batch_loss_history = []
final_predictions = []
final_labels = []

print("")
print("-- Making predictions on dev set with the best model --")
print("")


final_pred_start_time = time.time()

#make predictions using the best model weights
loop = tqdm(test_dataloader)
for sample in loop:
    sample['prediction'] = model(sample['X'])
    final_batch_accuracy_history.append((torch.argmax(sample['prediction'],1) == torch.argmax(sample['y'],1)).float().mean().item())
    final_batch_loss_history.append(loss_fn(sample['y'], sample['prediction']).item())

    #list of predictions and ground truths for confusion matrix
    final_predictions.extend(torch.argmax(sample['prediction'],1).detach().squeeze().tolist())
    final_labels.extend(torch.argmax(sample['y'],1).squeeze().tolist())

    #update dataframe with model predictions - used for preprocessing
    test_data.add_prediction(sample)

final_pred_stop_time = time.time()

print("-- Calculating accuracy and loss -- ")
print("")

#calculate and print final accuracies:
final_accuracy = mean(final_batch_accuracy_history)
final_loss = mean(final_batch_loss_history)
print(f"Model accuracy: {final_accuracy}")
print(f"Model loss: {final_loss}")



#save the dataframe with predictions for later use
test_data.df.to_csv(Path.repo + '/CNN/output_data/dev_CNN_preds.csv')

#Plots of train/dev accuracy/lost
plt.figure(0)
plt.plot(train_epoch_loss_history, label="train")
plt.plot(test_epoch_loss_history, label="dev")
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig(Path.repo + '/CNN/performance_figures/loss_hist.png')

plt.figure(1)
plt.plot(train_epoch_accuracy_history, label='train')
plt.plot(test_epoch_accuracy_history, label='dev')
plt.title(f"batch_size = {batch_size}, learning rate = {learning_rate}")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(Path.repo + '/CNN/performance_figures/accuracy_hist.png')


#calculate confusion matrix
CM = 0
CM += confusion_matrix(final_labels, final_predictions, labels=[0,1])
tn = CM[0][0]
tp = CM[1][1]
fp = CM[0][1]
fn = CM[1][0]

#calculate sensitivity + specificity
sensitivity, specificity = None, None
if ((tp+fn) == 0) or ((tn+fp) == 0):
    print("-------------")
    print("Could not calculate specificity/sensitivity. Zero Division Error. Consider balancing the dataset")
    print("-------------")
else:
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)


#print the calculated metrics
print("")
print("Here is the confusion matrix calculated by 'SKlearn")
print(CM)
print("")
print(f"True positives: {tp}")
print(f"False positives: {fp}")
print(f"True negatives: {tn}")
print(f"False negatives: {fn}")
print("")
if specificity:
    print(f"Specificity: {specificity}")
if sensitivity:
    print(f"Sensitivity: {sensitivity}")

#Plot and save the confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = CM, display_labels = [0, 1])
plt.figure(6)
cm_display.plot()
plt.savefig(Path.repo + '/CNN/performance_figures/confusion.png')


print("")
print("--Run time report--")
print("")

et = time.time()

print(f"Execution time: {(et-st)/60} minutes")
print(f"Average optimisation time: {mean(opt_times)} seconds")
print(f"Average prediction time: {mean(predd_times)} seconds")
print(f"Average test appending time: {test_append_time} seconds")
print(f"Time taken to make final prediction: {final_pred_stop_time - final_pred_start_time} seconds")
print("")
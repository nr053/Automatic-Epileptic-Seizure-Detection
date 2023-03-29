import pandas as pd
import torch
from torch import nn
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tqdm
import copy

path = "/Users/toucanfirm/Documents/DTU/Speciale/Sandbox/iris.data"

df = pd.read_csv(path, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])


#make numpy array of values
#turn into tensor



X = torch.tensor(df.iloc[:,0:4].values, dtype=torch.float32)
y = df.iloc[:,4:]

One_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
y = One_hot_encoder.transform(y)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # super initiation allows class easy class inheritance
        self.hidden = nn.Linear(4,8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8,3)

    def forward(self, x):
        x = self.hidden(x)
        x = self.act(x)
        x = self.output(x)

        return x
    
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.<span class="sig-name descname"><span class="pre">CrossEntropyLoss</span></span>()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

n_epochs = 200
batch_size = 5
batches_per_epoch = len(X_train) // batch_size

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

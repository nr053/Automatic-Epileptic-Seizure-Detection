"""Python module containing NN model classes"""

from torch import nn


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
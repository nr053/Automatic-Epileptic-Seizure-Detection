"""
Construct event times from model prediction to use for MNE annotations. 
"""

import tqdm

def train_loop(dataloader, model, loss_fn, optimizer):
    """Training loop"""

    model.train() # this should be placed outside the training loop

    batch_losses = []
    batch_accuracies = []

    loop = tqdm(dataloader)
    for sample in loop:
        logit_prediction = model(sample['X'])
        loss = loss_fn(logit_prediction, sample['y'])
        label_prediction = logit_prediction.round() #turn the probability into a binary prediction

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.mean().item()) # add the average loss over the batch 
        batch_accuracies.append((label_prediction == sample['y']).float().mean().item())    #add overage accuracy over the batch 

    return batch_losses.mean(), batch_accuracies.mean()



def test_loop(dataloader, model, loss_fn):  
    """An improved test loop that adds the model prediction to the dataframe"""

    batch_losses = []       # a list of the average loss of each batch
    batch_accuracies = []   # list holding the average accuracies of each batch

    model.eval()  #this should be activated outside the test loop to avoid it being called multiple times unnecessarily
    
    loop = tqdm(test_dataloader)
    for sample in loop:
        logit_prediction = model(sample['X'])
        sample['pred'] = logit_prediction               # add the prediction to the "pred" column of the dataframe. 
        loss = loss_fn(logit_prediction, sample['y'])
        label_prediction = logit_prediction.round()

        batch_losses.append(loss.mean().item()) 
        batch_accuracies.append((label_prediction == sample['y']).float().mean()).item()   

    return batch_losses.mean(),  batch_accuracies.mean()

            



def prediction_to_annotation(df_original):
    """Turn a dataframe of epoch paths along with their predictions to a dictionary with .edf file names and corresponding event times for MNE annotations"""
    df = df_original.loc[df_original.pred == 1].copy()
    df['epoch'] = df['path'].apply(lambda x: x.split('/')[-1]) #put epoch in separate column
    df['path'] = df['path'].apply(lambda x: '/'.join(x.split('/')[0:-1]) + ('.edf')) #remove epoch file tag and add ".edf"

    signals = []

    for recording in df['path'].unique():
        onset_times = df[df['path'] == recording]['epoch'].apply(lambda x: int(x.removesuffix('.pt'))).values
        recording = recording.replace("data_small", "TUSZ_V2/edf/train")
        signals.append(tuple((recording, onset_times)))

    return signals, df
"""
Extract the raw signal data from the .edf files and save them 
"""

import mne 
from p_tools import load_data, apply_montage, make_labels
import torch
import io
import time 
import os
import pandas as pd
from tqdm import tqdm

epoch_length = 1  # set duration of epochs 
#dict_list = []
#df = pd.DataFrame(index=range(2200000), columns=['path', 'label'])
df = pd.DataFrame(index=range(500000), columns=['path', 'label'])
st = time.time()

#with open('file_list_256.txt') as f:
with open('file_list_256_small.txt') as f:
    lines = [line.rstrip() for line in f]

# for file in lines[0]:
#     print(file)
#     edf_object, label_df = load_data(file)
#     bipolar_object = apply_montage(edf_object)
#     epochs = mne.make_fixed_length_epochs(bipolar_object, duration=epoch_length)
#     epoch_tensor = torch.tensor(epochs.get_data())
#     labels = make_labels(epoch_tensor, label_df)

#count = 0
curr_idx = 0

for line in tqdm(lines):
    #print(f"Files completed: {count}")
    #print(line)
    edf_object, label_df = load_data(line)
    bipolar_object = apply_montage(edf_object)
    epochs = mne.make_fixed_length_epochs(bipolar_object, duration=epoch_length)
    epoch_tensor = torch.tensor(epochs.get_data())
    labels = make_labels(epoch_tensor, label_df)


#    path = 'data/' + '/'.join(line.split('/')[9:]).removesuffix('.edf') + '/'
    path = 'data_small/' + '/'.join(line.split('/')[9:]).removesuffix('.edf') + '/'
    os.makedirs(path)

    print(time.time() - st)

    for j in range(len(epoch_tensor)):
        tmp_tensor = epoch_tensor[j].clone()
        file_name = path + str(j) + '.pt'
        torch.save(tmp_tensor, file_name)
        buffer = io.BytesIO()
        torch.save(tmp_tensor, buffer)
        
        #dict = {'path': file_name, 'label': int(labels[j][1])}
        #dict_list.append(dict)
        #df_tmp = pd.DataFrame({'epoch_path': file_name, 'label': int(labels[j][1])}, index=[count])
        df.loc[curr_idx] = [file_name, int(labels[j][1])]
        #df = pd.concat([df, df_tmp], ignore_index=True)
        curr_idx+=1

    #count += 1
    #if count == 5:
    #    break

#df = pd.DataFrame.from_dict(dict_list)
#df.to_csv(os.getcwd() + '/epoch_data.csv')
df = df.dropna()
df.to_csv(os.getcwd() + '/epoch_data_small.csv')

et = time.time()

print('Execution time: ', et-st, 'seconds')
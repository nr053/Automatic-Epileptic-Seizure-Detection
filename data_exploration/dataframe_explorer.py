"""
Given a csv of epoch .pt files, find the original recordings that contain seizures to run in the epoch_to_files.py script.

1. read csv file to dataframe
2. find the rows with seizure label = 1
3. remove the "epoch_number".pt extension to get the original .edf recording name
4. remove duplicates
5. add a ".edf" file extension and write to a file 
"""

import pandas as pd

df = pd.read_csv("/Users/toucanfirm/Documents/DTU/Speciale/tools/epoch_data.csv") #original data frame
df = df.dropna()  #drop empty rows
df = pd.DataFrame(df[df['label'] == 1]['path'].str.split('/')) #data frame holding list of path elements of files containing seizures
df['path'].apply(lambda x: x.pop()) #drop the epochs from the path elements. Now this is only the containing folders (unique recordings)
df['path'].apply(lambda x: x.pop(0)) #drop the "data" prefix from the path
df = df.drop_duplicates() #only unique recordings in the data frame
df['path'] = df['path'].apply(lambda x: '/'.join(x)) 
df['path'] = df['path'].apply(lambda x: '/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/' + x + '.edf') 


file_list = df['path'].values.tolist()

list_file = open('/Users/toucanfirm/Documents/DTU/Speciale/tools/file_list_256_small.txt' , 'w')

for file in file_list:
    print(file)
    list_file.write(file+"\n")


list_file.close()

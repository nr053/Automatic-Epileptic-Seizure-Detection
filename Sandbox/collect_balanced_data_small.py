"""Create a balanced data set from the epoch_data_small.csv data set."""


import pandas as pd
import numpy as np
import os

path = "/Users/toucanfirm/Documents/DTU/Speciale/tools/epoch_data_small.csv"

df = pd.read_csv(path)


df_seiz = df[df['label'] == 1]

df_bckg = df[df['label'] == 0]

seiz_index = df_seiz.index.values
bckg_index = df_bckg.sample(n = 83000, random_state = 0).index.values
indexes = np.concatenate((seiz_index, bckg_index))


df_final = df.filter(items = indexes, axis=0)
df_final = df_final.drop(df_final.columns[0], axis=1)
df_final.to_csv(os.getcwd() + '/balanced_data_small.csv')

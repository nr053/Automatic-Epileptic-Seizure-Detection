"""Display the distribution of seizure types across each of the three data splits (train/dev/eval)"""

import pandas as pd
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



df_train = pd.DataFrame()
df_dev = pd.DataFrame()
df_eval = pd.DataFrame()

for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf/train" + "/**/*.csv", recursive=True)):
    df = pd.read_csv(f, header=5)
    df_train = pd.concat([df_train, df])


for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf/dev" + "/**/*.csv", recursive=True)):
    df = pd.read_csv(f, header=5)
    df_dev = pd.concat([df_dev, df])

for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf/eval" + "/**/*.csv", recursive=True)):
    df = pd.read_csv(f, header=5)
    df_eval = pd.concat([df_eval, df])


df_train['duration'] = df_train['stop_time'] - df_train['start_time']


df_train['label'].value_counts().plot.bar(colormap='Paired')
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Train set class occurences')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/train_seizure_type_distribution.png")
df_dev['label'].value_counts().plot.bar()
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Dev set class occurences')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/dev_seizure_type_distribution.png")
df_eval['label'].value_counts().plot.bar()
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Eval set class occurences')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/eval_seizure_type_distribution.png")

df_train['duration'] = df_train['stop_time'] - df_train['start_time']
df_dev['duration'] = df_dev['stop_time'] - df_dev['start_time']
df_eval['duration'] = df_eval['stop_time'] - df_eval['start_time']

df_train.groupby('label')['duration'].sum().plot.bar()
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Train set class durations')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/train_seizure_type_duration_distribution.png")
df_dev.groupby('label')['duration'].sum().plot.bar()
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Dev set class durations')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/dev_seizure_type_duration_distribution.png")
df_eval.groupby('label')['duration'].sum().plot.bar()
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Eval set class durations')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/eval_seizure_type_duration_distribution.png")

#the background duration is so dominant that the scale of the other classes cannot be seen
#drop the background class and revisualise the bar chart


t_values = df_train.groupby(['label']).sum('duration')['duration'].values
t_index = df_train.groupby(['label']).sum('duration')['duration'].index.values
tt_values = np.delete(t_values, 1)
tt_index = np.delete(t_index, 1)

d_values = df_dev.groupby(['label']).sum('duration')['duration'].values
d_index = df_dev.groupby(['label']).sum('duration')['duration'].index.values
d_values = np.delete(d_values, 1)
d_index = np.delete(d_index, 1)

e_values = df_eval.groupby(['label']).sum('duration')['duration'].values
e_index = df_eval.groupby(['label']).sum('duration')['duration'].index.values
e_values = np.delete(e_values, 1)
e_index = np.delete(e_index, 1)


plt.bar(t_index, t_values)
plt.show()
plt.bar(tt_index, tt_values)
plt.show()


#df_train.groupby('label').sum().drop('bckg')['duration'].plot.bar()
df_t = df_train.groupby(by=['label']).sum('duration')['duration']
df_tt = df_t.drop('bckg')
df_tt.plot(kind='bar')
df_tt.plot(kind='bar')
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Train set class durations')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/train_seizure_type_duration_distribution_noBckg.png")
#df_dev.groupby('label').sum().drop('bckg')['duration'].plot.bar()
df_d = df_dev.groupby(by=['label']).sum('duration')['duration']
df_dd = df_d.drop('bckg')
df_dd.plot(kind='bar')
df_dd.plot(kind='bar')
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Dev set class durations')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/dev_seizure_type_duration_distribution_noBckg.png")
#df_eval.groupby('label').sum().drop('bckg')['duration'].plot.bar()
df_e = df_eval.groupby(by=['label']).sum('duration')['duration']
df_ee = df_e.drop('bckg')
df_e.plot(kind='bar')
df_e.plot(kind='bar')
plt.xlabel('seizure type')
plt.ylabel('occurences')
plt.title('Eval set class durations')
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/Plots/eval_seizure_type_duration_distribution_noBckg.png")
"""
collect all seizure durations in a dataframe and display distribution of durations in each set and in total.
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne
import torch
def load_data(file_name, epoch_length):
    raw_signal_object = mne.io.read_raw_edf(file_name, infer_types=True)
    timestamps_frame = pd.read_csv(file_name.removesuffix('edf') + 'csv_bi', header=5)

    bipolar_data = apply_montage(raw_signal_object)
    epochs = mne.make_fixed_length_epochs(bipolar_data, duration=epoch_length) # create epochs
    epoch_tensor = torch.tensor(epochs.get_data())
    annotated_data = write_annotations(bipolar_data, timestamps_frame)
    labels = make_labels(epoch_tensor, timestamps_frame)

    return bipolar_data, annotated_data, epoch_tensor, labels
    

def remove_suffix(word, suffixes):
    """Remove any suffixes contained in the 'suffixes' array from 'word'"""
    for suffix in suffixes:
        if word.endswith(suffix):
            return word.removesuffix(suffix)
    return word

def make_labels(epoch_tensor, df):
    """Creates class labels for the input data using the csv file"""
    labels = torch.empty([epoch_tensor.shape[0],2])
    for i,epoch in enumerate(labels):
        labels[i] = torch.tensor([1,0]) #set all class labels to [1,0] (background activity)
    for row in df.iterrows():
        if row[1]['label'] == "seiz":
            labels[round(row[1]['start_time']):round(row[1]['stop_time'])] = torch.tensor([0,1]) #set seizure class labels (round to nearest second)
    return labels

def calc_features(epoch_tensor):
    """This function calculates representative features of each channel in an epoch. 
    Each epoch is represented by 20 channels, which are represented by a list of features:
    
    1. Epoch mean
    2. Epoch variance
    3. Kurtosis
    4. """

    size = list(epoch_tensor.shape[:2])
    size.append(3)
    features = torch.empty(size)
    for i, epoch in enumerate(epoch_tensor):
        features[i] = torch.stack([epoch.mean(dim=1), epoch.var(dim=1), torch.tensor(kurtosis(epoch.numpy(),axis=1))],dim=1)
    return features

def apply_montage(data):
    """Apply the bipolar montage"""
    channel_renaming_dict = {name: remove_suffix(name, ['-LE', '-REF']) for name in data.ch_names}
    data.rename_channels(channel_renaming_dict)
    #print(data.ch_names)
    bipolar_data = mne.set_bipolar_reference(
        data.load_data(), 
        anode=['FP1', 'F7', 'T3', 'T5',
             'FP1', 'F3', 'C3', 'P3',
             'FP2', 'F4', 'C4', 'P4', 
            'FP2', 'F8', 'T4', 'T6',
            'T3', 'C3', 'CZ', 'C4'], 
        cathode=['F7','T3', 'T5', 'O1', 
                 'F3', 'C3', 'P3', 'O1', 
                 'F4', 'C4', 'P4', 'O2', 
                 'F8', 'T4', 'T6', 'O2',
                 'C3', 'CZ', 'C4', 'T4'], 
        drop_refs=True)
    bipolar_data.pick_channels(['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                            'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                            'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])
    return bipolar_data

def write_annotations(bipolar_data, labels_df):
    """Read annotations from csv file and write them to the mne object"""
    if labels_df['label'].eq("seiz").any(): #if the event df contains a seizure
        onset_times = labels_df['start_time'].values
        durations = labels_df['stop_time'].values - labels_df['start_time'].values 
        description = ["seizure"]
        annotations = mne.Annotations(onset_times, durations, description)
        bipolar_data.set_annotations(annotations)
    return bipolar_data

df_train = pd.DataFrame(columns=["set", "duration", "file"])

for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf/train" + '/**/*.csv_bi', recursive=True)):  #iterate through all *.csv_bi files")
    df = pd.read_csv(f, header=5)
    df = df.loc[df.label == 'seiz']
    df['set'] = "train"
    df['duration'] = df['stop_time'] - df['start_time']
    df['file'] = f

    df_train = pd.concat([df_train, df[['set', 'duration', 'file']]])



df_dev = pd.DataFrame(columns=["set", "duration", "file"])

for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf/dev" + '/**/*.csv_bi', recursive=True)):  #iterate through all *.csv_bi files")
    df = pd.read_csv(f, header=5)
    df = df.loc[df.label == 'seiz']
    df['set'] = "dev"
    df['duration'] = df['stop_time'] - df['start_time']
    df['file'] = f

    df_dev = pd.concat([df_dev,df[['set', 'duration', 'file']]])



df_eval = pd.DataFrame(columns=["set", "duration", "file"])

for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf/eval" + '/**/*.csv_bi', recursive=True)):  #iterate through all *.csv_bi files")
    df = pd.read_csv(f, header=5)
    df = df.loc[df.label == 'seiz']
    df['set'] = "eval"
    df['duration'] = df['stop_time'] - df['start_time']
    df['file'] = f

    df_eval = pd.concat([df_eval,df[['set', 'duration', 'file']]])


df_total = pd.concat([df_train, df_eval, df_dev])

df_train = df_train.reset_index()
df_dev = df_dev.reset_index()
df_eval = df_eval.reset_index()


print("------ Train Stats-------")
print("")
print(f"Maximum seizure length: {df_train.loc[df_train['duration'].idxmax()]}")
print(f"Minimum seizure length: {df_train.loc[df_train['duration'].idxmin()]}")
print(f"Mean seizure length: {df_train['duration'].mean()}")
print(f"Median seizure length: {df_train['duration'].median()}")
print("")
print("--------------------------")
print("------ Dev Stats-------")
print("")
print(f"Maximum seizure length: {df_dev.loc[df_dev['duration'].idxmax()]}")
print(f"Minimum seizure length: {df_dev.loc[df_dev['duration'].idxmin()]}")
print(f"Mean seizure length: {df_dev['duration'].mean()}")
print(f"Median seizure length: {df_dev['duration'].median()}")
print("")
print("--------------------------")
print("------ Eval Stats-------")
print("")
print(f"Maximum seizure length: {df_eval.loc[df_eval['duration'].idxmax()]}")
print(f"Minimum seizure length: {df_eval.loc[df_eval['duration'].idxmin()]}")
print(f"Mean seizure length: {df_eval['duration'].mean()}")
print(f"Median seizure length: {df_eval['duration'].median()}")
print("")
print("--------------------------")



plt.figure(0)
df_train.hist(column="duration", bins=100)
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/train_seizure_duration_distribution.png")

plt.figure(1)
df_dev.hist(column="duration", bins=100)
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/dev_seizure_duration_distribution.png")

plt.figure(2)
df_eval.hist(column="duration", bins=100)
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/eval_seizure_duration_distribution.png")

plt.figure(3)
df_total.hist(column="duration", bins=100)
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/data_exploration/total_seizure_duration_distribution.png")



#negative durations are obviously typos. Let's examine some of the files that say this. 

bipolar_data, annotated_data, epoch_tensor, labels = load_data(df_train.loc[df_train['duration'].idxmax()].file.removesuffix('csv_bi') + 'edf', 1)
annotated_data.plot()

bipolar_data, annotated_data, epoch_tensor, labels = load_data(df_train.loc[df_train['duration'].idxmin()].file.removesuffix('csv_bi') + 'edf', 1)
annotated_data.plot()

#start time of this seizure is 3141.6250, end time is 733

#let's remove the seizures with durations less than 0. and then 



bipolar_data, annotated_data, epoch_tensor, labels = load_data(df_eval.loc[df_eval['duration'].idxmin()].file.removesuffix('csv_bi') + 'edf', 1)
annotated_data.plot()


# negative seizure files. 
# array([[-2408.625,
#         '/home/migo/TUHP/TUSZ_V2/edf/train/aaaaajqo/s010_2010_05_04/03_tcp_ar_a/aaaaajqo_s010_t004.csv_bi'],
#        [-1388.0555999999997,
#         '/home/migo/TUHP/TUSZ_V2/edf/dev/aaaaahie/s018_2016_09_29/01_tcp_ar/aaaaahie_s018_t000.csv_bi'],
#        [-906.0,
#         '/home/migo/TUHP/TUSZ_V2/edf/dev/aaaaahie/s018_2016_09_29/01_tcp_ar/aaaaahie_s018_t000.csv_bi'],
#        [-810.9444000000003,
#         '/home/migo/TUHP/TUSZ_V2/edf/dev/aaaaahie/s018_2016_09_29/01_tcp_ar/aaaaahie_s018_t000.csv_bi'],
#        [-712.3888999999999,
#         '/home/migo/TUHP/TUSZ_V2/edf/dev/aaaaahie/s018_2016_09_29/01_tcp_ar/aaaaahie_s018_t000.csv_bi'],
#        [-341.0,
#         '/home/migo/TUHP/TUSZ_V2/edf/dev/aaaaahie/s018_2016_09_29/01_tcp_ar/aaaaahie_s018_t000.csv_bi']],
#       dtype=object)



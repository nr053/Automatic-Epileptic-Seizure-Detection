import pandas as pd
import mne 
import matplotlib.pyplot
import random
from parent.path import Path
from parent import p_tools
import torch
import matplotlib.pyplot as plt

df = pd.read_csv(Path.repo + '/DevEpochs/dev_only_records_with_seizures.csv')

# indexes = []

# for i in range(5):
#     indexes.append(random.randint(0, len(df)))

# for idx in indexes: 
#     epoch = torch.load(df.iloc[idx]['epoch']).detach()
#     label = df.iloc[idx]['gt']
#     raw = mne.io.read_raw_edf(df.iloc[idx]['recording'], infer_types=True, preload=True)
#     raw.filter(1,70)
#     raw.notch_filter(60)
#     bipolar = p_tools.apply_montage(raw)
#     bipolar.resample(250)

#     label_df = pd.read_csv(df.iloc[idx]['recording'].removesuffix('edf') + 'csv_bi', header=5)

#     if (label_df['label'].eq('seiz')).any():
#         for row in label_df.iterrows():
#             if row[1]['start_time'] <= df.iloc[idx]['timestamp']  < row[1]['stop_time']:
#                 annotation = 1
#     else:
#         annotation=0

#     bipolar.plot(n_channels=5, duration=2)
#     plt.plot(epoch[1])
#     plt.title(f"Annotation: {annotation}, gt_label: {df.iloc[idx]['gt']}")
#     plt.show()



idx = random.randint(0, len(df))

epoch = torch.load(df.iloc[idx]['epoch']).detach()
label = df.iloc[idx]['gt']
raw = mne.io.read_raw_edf(df.iloc[idx]['recording'], infer_types=True, preload=True)
raw.filter(1,70)
raw.notch_filter(60)
bipolar = p_tools.apply_montage(raw)
bipolar.resample(250)

label_df = pd.read_csv(df.iloc[idx]['recording'].removesuffix('edf') + 'csv_bi', header=5)

annotations = []

if (label_df['label'].eq('seiz')).any():
    for row in label_df.iterrows():
        if row[1]['start_time'] <= df.iloc[idx]['timestamp']  < row[1]['stop_time']:
            annotations.append(1)
        else:
            annotations.append(0)
else:
    annotations.append(0)


if 1 in annotations:
    annotation = 1
else:
    annotation = 0

plt.plot(epoch[1])
plt.title(f"Annotation: {annotation}, gt_label: {df.iloc[idx]['gt']}")
plt.show()

bipolar.plot(n_channels=5, duration=2, start=df.loc[idx,'timestamp'])



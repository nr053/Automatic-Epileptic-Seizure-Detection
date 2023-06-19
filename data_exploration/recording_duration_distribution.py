import mne
import pandas as pd
from parent.path import Path
from tqdm import tqdm
import glob

train_durs = []
dev_durs = []
eval_durs = []

for file in tqdm(glob.glob(Path.data + '/edf/train/**/*.edf', recursive=True)):
    raw = mne.io.read_raw_edf(file)
    duration = len(raw)/raw.info['sfreq']
    train_durs.append({'recording': file, 'duration': duration})

for file in tqdm(glob.glob(Path.data + '/edf/dev/**/*.edf', recursive=True)):
    raw = mne.io.read_raw_edf(file)
    duration = len(raw)/raw.info['sfreq']
    dev_durs.append({'recording': file, 'duration': duration})

for file in tqdm(glob.glob(Path.data + '/edf/eval/**/*.edf', recursive=True)):
    raw = mne.io.read_raw_edf(file)
    duration = len(raw)/raw.info['sfreq']
    eval_durs.append({'recording': file, 'duration': duration})


df_train = pd.DataFrame(train_durs)
df_dev = pd.DataFrame(dev_durs)
df_eval = pd.DataFrame(eval_durs)

print(f"Shortest recording durations in the train set: {df_train.sort_values('duration').head(10)}")
print(f"Shortest recording durations in the dev set: {df_dev.sort_values('duration').head(10)}")
print(f"Shortest recording durations in the eval set: {df_eval.sort_values('duration').head(10)}")
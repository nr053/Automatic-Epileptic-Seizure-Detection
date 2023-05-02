"""
This script iterates through all *.edf files in the data set and counts the channels used in each of the recordings.

Compare this with the results from find_montage.py

We need to use the same channels when applying the montage that are used in the *.csv files (if we want to do a multiclass problem).
"""

import glob
import mne 
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def remove_suffix(word, suffixes):
    """Remove any suffixes contained in the 'suffixes' array from 'word'"""
    for suffix in suffixes:
        if word.endswith(suffix):
            return word.removesuffix(suffix)
    return word

count = Counter()

for f in tqdm(glob.glob("/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf" + '/**/*.edf', recursive=True)):
    data = mne.io.read_raw_edf(f, infer_types=True)

    channel_renaming_dict = {name: remove_suffix(name, ['-LE', '-REF']) for name in data.ch_names}
    data.rename_channels(channel_renaming_dict)

    count.update(Counter(data.ch_names))


labels, values = zip(*count.items())

indexes = np.arange(len(labels))
width = 0.9

plt.barh(indexes, values, width, align='center', alpha=0.9)
plt.yticks(indexes, labels)
plt.xlabel('No. of occurences')
plt.show()


df = pd.DataFrame.from_dict(count, orient='index')
print(df.sort_values(0, ascending=False)).head(30)
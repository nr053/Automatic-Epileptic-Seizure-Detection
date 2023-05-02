"""
This script is to iterate through all the *.csv files in the data set and find the channels being used in the multiclass annotation file.

This allows to make the decision whether one montage can be applied across all recordings. 
"""


import pandas as pd
import glob
from collections import Counter 
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

count = Counter()

for f in tqdm(glob.glob("/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf" + "/**/*.csv", recursive=True)):
    df = pd.read_csv(f, header=5)

    assert list(df.columns) == ['channel', 'start_time', 'stop_time', 'label', 'confidence'], f"dataframe columns do not match for file: {f}"

    count.update(Counter(df['channel']))





print(count)

labels, values = zip(*count.items())

indexes = np.arange(len(labels))
width = 0.9

plt.barh(indexes, values, width, align='center', alpha=0.9)
plt.yticks(indexes, labels)
plt.xlabel('No. of occurences')
plt.show()

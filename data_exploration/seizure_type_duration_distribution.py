"""Inspect and visualise the distribution of seizure durations for each seizure type."""

import pandas as pd
from tqdm import tqdm
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame()

for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf" + "/**/*.csv", recursive=True)):
    if f != '/home/migo/TUHP/TUSZ_V2/edf/dev/aaaaahie/s018_2016_09_29/01_tcp_ar/aaaaahie_s018_t000.csv' and f != '/home/migo/TUHP/TUSZ_V2/edf/train/aaaaajqo/s010_2010_05_04/03_tcp_ar_a/aaaaajqo_s010_t004.csv':
        df_tmp = pd.read_csv(f, header=5)
        df_tmp['filename'] = f
        df = pd.concat([df, df_tmp])


df['duration'] = df['stop_time'] - df['start_time']

df_gnsz = df[df['label'] == "gnsz"]
df_fnsz = df[df['label'] == "fnsz"]
df_cpsz = df[df['label'] == "cpsz"]
df_absz = df[df['label'] == "absz"]
df_spsz = df[df['label'] == "spsz"]
df_tcsz = df[df['label'] == "tcsz"]
df_tnsz = df[df['label'] == "tnsz"]
df_mysz = df[df['label'] == "mysz"]


df_gnsz.hist(column="duration", bins=100)
df_fnsz.hist(column="duration", bins=100)
df_cpsz.hist(column="duration", bins=100)
df_absz.hist(column="duration", bins=100)
df_spsz.hist(column="duration", bins=100)
df_tcsz.hist(column="duration", bins=100)
df_tnsz.hist(column="duration", bins=100)
df_mysz.hist(column="duration", bins=100)
plt.show()


# fig, ax = plt.subplots()

# gnsz_heights, gnsz_bins = np.histogram(df_gnsz['duration'])
# fnsz_heights, fnsz_bins = np.histogram(df_fnsz['duration'])
# cpsz_heights, cpsz_bins = np.histogram(df_cpsz['duration'])
# absz_heights, absz_bins = np.histogram(df_absz['duration'])
# spsz_heights, spsz_bins = np.histogram(df_spsz['duration'])
# tcsz_heights, tcsz_bins = np.histogram(df_tcsz['duration'])
# tnsz_heights, tnsz_bins = np.histogram(df_tnsz['duration'])
# mysz_heights, mysz_bins = np.histogram(df_mysz['duration'])

# width = (gnsz_bins[1] - gnsz_bins[0])/3

# ax.bar(gnsz_bins[:-1], gnsz_heights, width=width, facecolor='cornflowerblue')
# ax.bar(fnsz_bins[:-1]+width, fnsz_heights, width=width, facecolor='seagreen')
# ax.bar(cpsz_bins[:-1], cpsz_heights, width=width, facecolor='cornflowerblue')
# ax.bar(absz_bins[:-1]+width, abszsz_heights, width=width, facecolor='seagreen')
# ax.bar(spsz_bins[:-1], spsz_heights, width=width, facecolor='cornflowerblue')
# ax.bar(tcsz_bins[:-1]+width, fnsz_heights, width=width, facecolor='seagreen')
# ax.bar(tnsz_bins[:-1], gnsz_heights, width=width, facecolor='cornflowerblue')
# ax.bar(mysz_bins[:-1]+width, fnsz_heights, width=width, facecolor='seagreen')

plt.figure(figsize=(8,6))
#plt.hist(df_gnsz['duration'], bins=100, alpha=0.5, label="gnsz")
#plt.hist(df_fnsz['duration'], bins=100, alpha=0.5, label="fnsz")
plt.hist(df_cpsz['duration'], bins=100, alpha=0.5, label="cpsz")
plt.hist(df_absz['duration'], bins=100, alpha=0.5, label="absz")
plt.hist(df_spsz['duration'], bins=100, alpha=0.5, label="spsz")
plt.hist(df_tcsz['duration'], bins=100, alpha=0.5, label="tcsz")
plt.hist(df_tnsz['duration'], bins=100, alpha=0.5, label="tnsz")
plt.hist(df_mysz['duration'], bins=100, alpha=0.5, label="mysz")
plt.xlabel("Duration", size=14)
plt.ylabel("Count", size=14)
plt.title("Distribution of seizure durations")
plt.legend(loc='upper right')
plt.show()

#there are only a handful of seizure longer than 1500 seconds. Remove these to get a proper look at the distribution
"""
Count the number of seizure durations less than two seconds
"""

import glob
import pandas as pd
from tqdm import tqdm


df = pd.DataFrame(columns=["channel","start_time","stop_time","label","confidence"])

for f in tqdm(glob.glob("/home/migo/TUHP/TUSZ_V2/edf/" + '/**/*.csv_bi', recursive=True)):  #iterate through all *.csv_bi files")
    df_tmp = pd.read_csv(f, header=5)
    df_tmp = df_tmp.loc[df_tmp.label == 'seiz']
    df_tmp['duration'] = df_tmp['stop_time'] - df_tmp['start_time']

    df = pd.concat([df, df_tmp])

df[df['duration'].between(0,2)].head()

print(f"Number of seizures less than 2 seconds is {len(df[df['duration'].between(0,2)])}")


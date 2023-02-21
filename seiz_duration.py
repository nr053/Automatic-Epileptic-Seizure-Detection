"""
1. get *.csv_bi files
2. remove files that do not contain "seiz"
3. find start and end time of seizure event
4. calculate seizure duration
"""

import glob
import pandas as pd
import argparse
import os

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--set", type=str, help="the set on which the operation should be performed" )
args = argParser.parse_args()

path = os.getcwd() + '/TUSZ_v2/edf/' + args.set #set path
total_seizure_duration = 0


for f in glob.glob(path + '/**/*.csv_bi', recursive=True):
    #print(f)
    file = open(f)
    df = pd.read_csv(f, header=5)
    #print(df)
    #print(df.dtypes)
    if df['label'].eq("seiz").any():
        df['duration'] = df['stop_time'] - df['start_time']
        #print(df)
        duration = df['duration'].sum()
        #print("duration = ", duration)
        total_seizure_duration += duration

print("Total seizure duration: ", total_seizure_duration)

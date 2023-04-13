"""
1. get *.csv_bi files
2. remove files with no seizures
3. find total file duration
"""

import glob
import pandas as pd
import argparse
import os

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--set", type=str, help="the set to perform operation on")
args = argParser.parse_args()

path = os.getcwd() + '/TUSZ_V2/edf/' + args.set #set path
total_file_duration = 0

for f in glob.glob(path + '/**/*.csv_bi', recursive=True):  #iterate through all *.csv_bin files
    file = open(f)  #open file
    df = pd.read_csv(file, names=['a', 'b', 'c', 'd', 'e']) #read csv into dataframe with columns "a,b,c,d,e"

    if df['d'].eq("seiz").any(): #if there is a seizure event in the label column
        dur = df.loc[2,'a'] #locate the duration field
        str_list = dur.split() #split the string into a str list
        duration = float(str_list[3])   #access the duration
        total_file_duration += duration #sum the durations

print("total duration of files with seizures: ",total_file_duration)
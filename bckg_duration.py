"""
1. get *.csv_bi files
2. sum seizure times
3. remove from total duration
"""

import glob
import pandas as pd
import argparse
import os
from path import Path

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--set", type=str, help="the set on which the operation should be performed")
args = argParser.parse_args()

path = Path.ext_path + args.set #set path
total_seizure_duration = 0
total_duration = 0


for f in glob.glob(path + '/**/*.csv_bi', recursive=True):  #iterate through all *.csv_bi files
    print(f)    
    file = open(f)  #open the file
    df_events = pd.read_csv(f, header=5)    #read events into a data frame
    df_duration = pd.read_csv(f, nrows=1, skiprows=2, names=['a'])  #read the file duration row into another dataframe
    df_duration['a'] = df_duration['a'].astype("string")    #convert file duration field to a string
    str = df_duration.loc[0].at['a']    #extract file duration string
    str_list = str.split() #split the string into a str list
    total_file_duration = float(str_list[3])    #extract file duration as a float
    total_duration += total_file_duration
    if df_events['label'].eq("seiz").any(): #if the event df contains a seizure
        df_events['duration'] = df_events['stop_time'] - df_events['start_time']    #add a seizure duration column
        seizure_duration = df_events['duration'].sum()  #sum the file seizure durations
        print(df_events)
        #print("seizure duration = ", seizure_duration)  
        total_seizure_duration += seizure_duration  #add the file seizure duration to the total seizure time




print("Total seizure duration: ", total_seizure_duration)
print("Total duration: ", total_duration)
print("Total background duration: ", total_duration - total_seizure_duration)




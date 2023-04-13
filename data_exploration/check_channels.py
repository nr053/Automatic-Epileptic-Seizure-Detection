"""
This script takes a list of files and outputs the channels + counts 
that are used throughout the recording. 

Extended to include the duration of all files vs the duration of files 
containing aurical electrodes.

1. replace .edf with .csv_bi in each file suffix
2. read csv
3. sum duration
"""

import mne 
import pandas as pd
from collections import Counter
from path import Path
import argparse

# argParser = argparse.ArgumentParser()
# argParser.add_argument("-f", "--folder_location", choices = ['local', 'external'], tpye=str, help="local or external folder location")
# args = argParser.parse_args()

# if args.folder_location == 'local':
#     path = Path.machine_path
# else:
#     path = Path.ext_path


def count_channels(file):
    """Counts the occurence of each channel in the recordings"""
    channel_set = set()
    cnt = Counter()
    aurical_list = []
    with open(file, 'r') as f:
        for line in f.readlines():
            #print(line.strip())
            data = mne.io.read_raw_edf(line.strip(), infer_types=True)

            for chName in data.info['ch_names']:
                channel_set.add(chName)
                cnt[chName] += 1

            if any("A1-REF" in s for s in data.info['ch_names']):
                aurical_list.append(line.strip())
    return cnt, aurical_list
        

def sum_durations(file_list, aurical_file_list):
    overall_duration = 0
    aurical_duration = 0
    with open(file_list, 'r') as f:
        for line in f.readlines():
            csv_file = line.strip().removesuffix(".edf") + ".csv_bi"

            df_duration = pd.read_csv(csv_file, names=['a'], nrows=1, skiprows=2)
            str = df_duration['a'].loc[0]
            str_list = str.split()
            duration = float(str_list[3])
            #str_list = float(df_duration['a'].loc[0].split()[3])
            overall_duration += duration

        for file in aurical_file_list:
            csv_file = file.removesuffix(".edf") + ".csv_bi"
            df_duration = pd.read_csv(csv_file, names=['a'], nrows=1, skiprows=2)
            str = df_duration['a'].loc[0]
            str_list = str.split()
            duration = float(str_list[3])
            aurical_duration += duration

    return overall_duration, aurical_duration
            
def main(file):
    cnt, aurical_list = count_channels(file)
    print(aurical_list)
    overall_duration, aurical_duration = sum_durations(file, aurical_list)
    print(cnt)
    print("Recording duration across all 3107 files: ", overall_duration)
    print("Recording duration across files containing 'A1': ", aurical_duration)
    return

if __name__ == "__main__":
    main("file_list.txt")

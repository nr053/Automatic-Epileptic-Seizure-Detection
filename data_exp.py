"""Data exploration script

Find:

1. number of patients
2. number of recording sessions
3. number of recordings
4. number of seizure events
5. seizure duration
6. background duration
7. total recording duration

8. filter the recordings by sampling rate (multiple of 256/512 Hz)

9. repeat steps 1-7 
"""

import mne
import os
import glob
import pandas as pd
import argparse
from path import Path
from collections import Counter

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--set", default= "", type=str, 
                       help="the set on which the operation should be performed")
argParser.add_argument("-p", "--path", type=str, 
                       choices=["local", "ext"],
                       help="the path location of the parent directory (machine/ext hard drive)")
args = argParser.parse_args()

if args.path == "local":
    path = Path.machine_path + args.set 
else:
    path = Path.ext_path + args.set 


def collect_files(path):
    """Collect a list of edf files and a list of csv files to calculate stats on"""
    edf_list = []
    csv_list = []
    for f in glob.glob(path + '/**/*.edf', recursive=True):
        edf_list.append(f)
    for f in glob.glob(path + '/**/*.csv_bi', recursive=True):
        csv_list.append(f)
    
    return edf_list, csv_list

def diff_list(edf_list, csv_list):
    edf_list = [edf.removesuffix('.edf') for edf in edf_list]
    csv_list = [csv.removesuffix('.csv_bi') for csv in csv_list]

    return list(set(csv_list) - set(edf_list))
    
def count_recordings(edf_list):
    patient_set = set()
    session_set = set()

    list = [file.split("/") for file in edf_list]
    for arr in list:
        patient_set.add(arr[9])
        session_set.add(arr[9] + '/' + arr[10])
    
    return patient_set, session_set

def calculate_durations(csv_list):
    total_seizure_duration = 0
    total_duration = 0
    for csv_file in csv_list:
        #print("CSV FILE TO BE READ: ", csv_file)
        df_events = pd.read_csv(csv_file, header=5)    #read events into a data frame
        df_duration = pd.read_csv(csv_file, nrows=1, skiprows=2, names=['a'])  #read the file duration row into another dataframe
        df_duration['a'] = df_duration['a'].astype("string")    #convert file duration field to a string
        str = df_duration.loc[0].at['a']    #extract file duration string
        str_list = str.split() #split the string into a str list
        total_file_duration = float(str_list[3])    #extract file duration as a float
        total_duration += total_file_duration
        if df_events['label'].eq("seiz").any(): #if the event df contains a seizure
            df_events['duration'] = df_events['stop_time'] - df_events['start_time']    #add a seizure duration column
            seizure_duration = df_events['duration'].sum()  #sum the file seizure durations
            #print(df_events)
            #print("seizure duration = ", seizure_duration)  
            total_seizure_duration += seizure_duration  #add the file seizure duration to the total seizure time

    return  total_duration, total_seizure_duration

def sfreq_subset(edf_list):
    """Takes a list of edf files and returns:

    1. list of records sampled at 256/512Hz
    2. list of records sampled at 256/512Hz containing aurical channels 
    3. count of channels used 
    4. count of channels used in recordings sampled at 256/512Hz. 
    """
    sfreq_list = []
    aurical_list = []
    #usable_list = []
    #channel_set = set()
    cnt_channels = Counter()
    #channel_sfreq_set = set()
    cnt_channels_sfreq = Counter()
    for edf_file in edf_list:
        #print("FIRST EDF FILE: ", edf_file)
        data = mne.io.read_raw_edf(edf_file, infer_types=True)
        for chName in data.info['ch_names']:
            chName = chName.removesuffix('-LE')
            chName = chName.removesuffix('-REF')
            #channel_set.add(chName)
            cnt_channels[chName] += 1
        if data.info['sfreq'] % 256 == 0:
            sfreq_list.append(edf_file)
            for chName in data.info['ch_names']:
                chName = chName.removesuffix('-LE')
                chName = chName.removesuffix('-REF')
                #channel_sfreq_set.add(chName)
                cnt_channels_sfreq[chName] += 1

                if chName == 'A1':
                    aurical_list.append(edf_file)
    
    return sfreq_list, aurical_list, cnt_channels, cnt_channels_sfreq 


            



    return sfreq_list, cnt_channels, cnt_channels_sfreq 

def count_seizures(csv_list):
    count = 0
    for file in csv_list:
        #print("THE FILE IN QUESTION IS: ", file)
        df = pd.read_csv(file, header=5)
        #print("DATAFRAME: ", df)
        #c = df['label'].value_counts()
        #print("COUNTS: ", c)
        #num_seizures = c['seiz']
        num_seizures = df['label'].str.count('seiz').sum()
        count += num_seizures
    
    return count


# def count_channels(edf_list):
#     """Counts the occurence of each channel in the recordings"""
#     channel_set = set()
#     cnt = Counter()
#     aurical_list = []
#     for edf_file in edf_list:
#         #print(line.strip())
#         data = mne.io.read_raw_edf(edf_file, infer_types=True)
#         for chName in data.info['ch_names']:
#             channel_set.add(chName)
#             cnt[chName] += 1

#         if any("A1-REF" in s for s in data.info['ch_names']):
#             aurical_list.append(edf_file)
    
#     return cnt, aurical_list
        


def main():
    edf_list, csv_list = collect_files(path)

    list_diff = diff_list(edf_list, csv_list)
    
    #print("HERE IS YOUR CSV LIST MOFO: ", csv_list)
    total_duration, total_seizure_duration = calculate_durations(csv_list)
    seizure_count = count_seizures(csv_list)

    sfreq_edf_list, aurical_list, cnt_channels, cnt_channels_sfreq  = sfreq_subset(edf_list)
    sfreq_csv_list = [os.path.splitext(i)[0]+'.csv_bi' for i in sfreq_edf_list] #convert edf files to their corresponding csv_bi files
    aurical_csv_list = [os.path.splitext(i)[0]+'.csv_bi' for i in aurical_list]

    total_sfreq_duration, total_sfreq_seizure_duration = calculate_durations(sfreq_csv_list)
    total_aurical_duration, total_aurical_seizure_duration = calculate_durations(aurical_csv_list)

    seizure_sfreq_count = count_seizures(sfreq_csv_list)

    patient_set, session_set = count_recordings(edf_list)
    patient_sfreq_set, session_sfreq_set = count_recordings(sfreq_edf_list)

    #cnt, aurical_list = count_channels(edf_list)
    #cnt_sfreq, aurical_sfreq_list = count_channels(sfreq_edf_list)

    #print(patient_set)
    #print(session_set)

    
    # open file in write mode
    with open(r'usable_edf_files.txt', 'w') as fp:
        for file in sfreq_edf_list:
            # write each item on a new line
            fp.write("%s\n" % file)

    print("PATH IS ::::::  ", path)

    print("differences between edf and csv lists: ", list_diff)
    
    print("number of edf files: ", len(edf_list))
    print("number of csv_files: ", len(csv_list))

    print("number of patients: ", len(patient_set))
    print("number of sessions: ", len(session_set))

    print("number of seizure events: ", seizure_count)

    print("Total seizure duration: ", total_seizure_duration)
    print("Total background duration: ", total_duration - total_seizure_duration)
    print("Total duration: ", total_duration)

    print("Used channels: ", cnt_channels)
    #print("Aurical channels: ", aurical_list)



    print("number of patients sampled at 256/512Hz: ", len(patient_sfreq_set))
    print("number of sessions sampled at 256/512Hz: ", len(session_sfreq_set))
    print("number of recordings sampled at 256/512Hz: ", len(sfreq_edf_list))

    print("number of seizure events sampled at 256/512Hz: ", seizure_sfreq_count)

    print("Total seizure duration of recordings at 256/512Hz: ", total_sfreq_seizure_duration)
    print("Total background duration of recordings at 256/512Hz: ", total_sfreq_duration - total_sfreq_seizure_duration)
    print("Total duration of recordings at 256/512Hz: ", total_sfreq_duration)

    print("Used channels recorded at 256/512Hz: ", cnt_channels_sfreq)
    #print("Aurical channels recorded at 256/512Hz: ", aurical_sfreq_list)
    print("")
    print("Total seizure duration of recordings sampled at 256/512Hz containing aurical channels: ", total_aurical_seizure_duration)
    print("Total duration of recordings sampled at 256/512Hz containing aurical channels: ", total_aurical_duration)

    return

if __name__ == "__main__":
    main()









"""This script extracts the seizure start and stop times from the .csv_bi file
and creates an array to hold the class label for each epoch.

1. Extract seizure start/stop times
2. create array size of (total duration)/(epoch duration)
3. These should be done in seperate functions since the file openings seem to be
messing with each other. 
 """

import csv
import pandas as pd
import numpy as np

path = '/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaabhz/s003_2010_09_01/03_tcp_ar_a'


def label_extractor(path):
    file = open(path + '/aaaaabhz_s003_t001.csv_bi')
    df = pd.read_csv(file, header=5)
    print(df.head())

    seiz_times = []
    print("\n------------Here's a line breaker--------------")
    for index, row in df.iterrows():
        print(index, row['start_time'], row['stop_time'])
        seiz_times.append([row['start_time'], row['stop_time']])

    print("\n--------ready for the seizure times????---------")

    print("seiz_times: ", seiz_times)

    return seiz_times


def duration_extractor(path):
    file = open(path + '/aaaaabhz_s003_t001.csv_bi')
    data = list(csv.reader(file))
    #dur = data[2].split()
    duration = float(data[2][0].split()[3])

    print(duration)
    return duration

seizure_times = label_extractor(path)
recording_duration = duration_extractor(path)

#seizure times extractor
#total duration extractor
#round seizure times to nearest multiple of epoch (nearest second/half second etc.)
#create label vector with size = no of epochs
#assign label class to each epoch.

epoch_duration = 1
num_epochs = recording_duration/epoch_duration

#seiz_start_times = [round(start_time) for start_time in seiz_start_times]
#seiz_stop_times = [round(stop_time) for stop_time in seiz_stop_times]

seizure_times = [[round(start), round(stop)] for [start, stop] in seizure_times]
print(seizure_times)

epoch_labels = np.zeros(int(num_epochs))

for [start, stop] in seizure_times:
    for i in range(len(epoch_labels)):
        if start < i < stop:
            epoch_labels[i] = 1

print(epoch_labels)
print(len(epoch_labels))


count = 0
for i in range(len(epoch_labels)):
    if epoch_labels[i] != epoch_labels[i-1]: 
        print(i)
        

"""this is a short script to ensure that all the 
*.csv_bi files are formatted correctly for my purpose.
The seizure details begin on line 6.

Running this script gives 7377 files conforming to this format with 0 exceptions.
Query 'find TUSZ_V2/edf -name "*.csv_bi" | wc' gives 7377 total .csv_bi files. 
So all files conform. 
"""

import glob
import csv

path = '/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf'
correct = 0
incorrect = 0 

header = ['channel', 'start_time', 'stop_time', 'label', 'confidence']

for f in glob.glob(path + '/**/*.csv_bi', recursive=True):
    file = open(f, "r")
    data = list(csv.reader(file))
    #print(f)
    #print(data[5])
    print(data[2])
    if data[5] == header:
        correct += 1
    else:
        incorrect += 1
    #print(correct)

print("Correct: ", correct)
print("Incorrect: ", incorrect)
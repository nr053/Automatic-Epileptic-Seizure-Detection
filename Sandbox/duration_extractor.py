"""This script extracts the total duration of a recording to split into epochs"""

import csv
import pandas as pd

path = '/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaabhz/s003_2010_09_01/03_tcp_ar_a' 
file = open(path + '/aaaaabhz_s003_t001.csv_bi')
data = list(csv.reader(file))
#dur = data[2].split()
duration = float(data[2][0].split()[3])

print(duration)
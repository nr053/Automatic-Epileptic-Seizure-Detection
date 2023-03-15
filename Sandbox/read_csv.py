"""This script reads a .csv_bi file and prints the rows, and recording duration. """

import csv

path = '/Users/toucanfirm/Documents/DTU/Speciale'

file = open(path + '/TUSZ_V2/edf/train/aaaaaaac/s001_2002_12_23/02_tcp_le/aaaaaaac_s001_t000.csv_bi')
csvreader = csv.reader(file)

rows = []
for row in csvreader:
    rows.append(row)
print(rows)
file.close()


n = len(rows)
last_row = rows[n-1]
print("last row: ", last_row)
start_time = float(last_row[1])
end_time = float(last_row[2])

duration = end_time - start_time
print(duration)
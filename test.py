"""
A small script to test the visualisations on Rose's server
"""

import mne
import matplotlib.pyplot as plt

file_path = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf"

data = mne.io.read_raw_edf(file_path)
data.plot()
plt.savefig("test_plot.png")
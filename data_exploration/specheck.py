import mne
import matplotlib.pyplot as plt 
from parent.path import Path

file = Path.data + '/edf/train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf'

raw = mne.io.read_raw_edf(file, infer_types=True, preload=True)

data = raw.get_data()

plt.specgram(data[0], Fs=raw.info['sfreq'])
plt.savefig('/home/migo/TUHP/specheck.png')

raw.filter(l_freq=1, h_freq=70)
raw.notch_filter(freqs=60)
data_filtered = raw.get_data()
plt.specgram(data_filtered[0], Fs=raw.info['sfreq'])
plt.savefig('/home/migo/TUHP/specheck_filtered.png')


data_direct_filter = mne.filter.filter_data(data, sfreq=raw.info['sfreq'], l_freq=1, h_freq=70)
plt.specgram(data_direct_filter[0], Fs=raw.info['sfreq'])
plt.savefig('/home/migo/TUHP/specheck_direct_filtered.png')


# raw = mne.io.read_raw_edf(file, infer_types=True, preload=True)

# data = raw.get_data()

# plt.specgram(data[0], Fs=raw.info['sfreq'])
# plt.savefig('/home/migo/TUHP/specheck.png')

# raw_filtered = raw.filter(l_freq=1, h_freq=70)
# raw_notched = raw_filtered.notch_filter(freqs=60)
# plt.specgram(data[0], Fs=raw.info['sfreq'])
# plt.savefig('/home/migo/TUHP/specheck_filtered.png')




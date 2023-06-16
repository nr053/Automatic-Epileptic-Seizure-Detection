# -*- coding: utf-8 -*-
"""
An example where effecting jittering of triggers occurs when
downsampling before epoching.
"""
import numpy as np
import matplotlib.pyplot as plt

# 1 sec of data @ 1000 Hz
fs = 1000.  # Hz
decim = 5
n_samples = 1000
freq = 20.  # Hz
t = np.arange(n_samples) / fs
epoch_dur = 1. / freq  # 2 cycles of our sinusoid

# we have a 10 Hz sinusoid signal
raw_data = np.cos(2 * np.pi * freq * t)

# let's make events that should show the sinusoid moving out of phase
# continuously
n_events = 40
event_times = np.linspace(0, 1. / freq, n_events, endpoint=False)
event_samples = np.round(event_times * fs).astype(int)
data_epoch = list()
epoch_len = int(round(epoch_dur * fs))
for event_time in event_times:
    start_idx = int(np.round(event_time * fs))
    data_epoch.append(raw_data[start_idx:start_idx + epoch_len])
data_epoch = np.array(data_epoch)
data_epoch_ds = data_epoch[:, ::decim]

# now let's try downsampling the raw data instead
raw_data_ds = raw_data[::decim]
fs_new = fs / decim
data_ds_epoch = list()
epoch_ds_len = int(round(epoch_dur * fs_new))
for event_time in event_times:
    start_idx = int(np.round(event_time * fs_new))
    data_ds_epoch.append(raw_data_ds[start_idx:start_idx + epoch_ds_len])
data_ds_epoch = np.array(data_ds_epoch)

# Look at the results
assert data_ds_epoch.shape == data_epoch_ds.shape
fig, axs = plt.subplots(1, 2)
t_ds = np.arange(epoch_ds_len) / fs_new
for di, (data_e_d, data_d_e) in enumerate(zip(data_epoch_ds, data_ds_epoch)):
    color = [di / float(n_events + 10)] * 3
    axs[0].plot(t_ds, data_e_d, color=color)
    axs[1].plot(t_ds, data_d_e, color=color)
axs[0].set_ylabel('Epoch then downsample')
axs[1].set_ylabel('Downsample then epoch')
for ax in axs:
    ax.set_xlim(t_ds[[0, -1]])
    ax.set_xticks(t_ds[[0, -1]])
fig.set_tight_layout(True)

plt.show()
"""
Use the raw signal values in a neural network to predict the activity type.

1. read edf file
2. apply montage (+ HP/LP filters)
3. split into epochs
4. assign classes 
5. train model 
6. predict with model
7. assess performance
"""

import glob
import mne 
from path import Path

for f in glob.glob(Path.local_path + '**/*.edf', recursive=True):
    data = mne.io.read_raw_edf(f, infer_types=True)
    if data.info['sfreq'] == 256:

    

def downsampling(data):
    """low-pass filter, epoch, decimate"""
    curr_sfreq = data.info['sfreq']
    new_sfreq = 256 # Hz
    decim = np.round(curr_sfreq / new_sfreq).astype(int)
    lowpass_freq = 70
    assert lowpass_freq <= new_sfreq/3
    data_filtered = data.copy().filter(l_freq=None, h_freq=lowpass_freq)
    #events = mne.find_events(data_filtered)
    duration = data.n_times / curr_sfreq
    epoch_dur = 1
    n_epochs = duration / epoch_dur
    
    events = np.zeros([int(n_epochs), 3])
    for i in range(int(n_epochs)):
        events[i][0] = i
        events[i][2] = i+1 
    events = events.astype(int)
    epochs = mne.Epochs(data_filtered, events, decim = decim)

    return data_filtered, events, epochs
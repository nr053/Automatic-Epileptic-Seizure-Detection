"""
- Select a file with 512 sampling rate
- compare downsampling techniques
- visualise epoch
"""

import mne 
import glob
import matplotlib.pyplot as plt
import numpy as np

path = '/Volumes/Kingston/TUSZ_V2/edf/train/'

def find_512_Hz_recording(path):
    """This function finds an .edf recorded at 512 Hz sampling frequency."""
    for f in glob.glob(path + '/**/*.edf', recursive=True):
        data = mne.io.read_raw_edf(f, infer_types = True, preload=True)
        if data.info['sfreq'] == 512:
            break
    
    return f, data



def resampling(data):
    """Uses raw.resample() to downsample by x0.5"""    
    #raw.resample(). Generally not recommended as downsanpling raw data is prone to jittering. 
    resampled = data.copy().resample(256)
    
    return data, resampled

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



def plot_psd(data, resampled):
    """
    Compares the power spectral density plots of original and downsampled data
    """
    for data, title in zip([data, resampled], ['Original', 'Downsampled']):
        fig = data.compute_psd().plot()
        fig.subplots_adjust(top=0.9)
        fig.suptitle(title)
        plt.ylim([0,60])
        plt.setp(fig.axes)
        print(plt.gca())

def show_recording(data, resampled):
    data.plot(duration=5, n_channels=34)
    resampled.plot(duration=5, n_channels=34)


def main():
    file, raw = find_512_Hz_recording(path)
    print(file)
    data, resampled = resampling(raw)
    print("data info: ", data.info)
    print("resampled info: ", resampled.info)
    plot_psd(data, resampled)
    show_recording(data, resampled)
    data_filtered, events, epochs = downsampling(raw)
    print(data_filtered.info)
    print(events)
    print(epochs)
    plot_psd(raw, data_filtered)
    show_recording(raw, data_filtered)
    return

if __name__ == "__main__":
    main()


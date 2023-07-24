"""Epoch continuous data and visualize."""

import mne
import pandas as pd
import numpy as np
from p_tools import remove_suffix
from path import Path
import matplotlib.pyplot as plt
from scipy import signal

#file = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaartn/s007_2014_11_09/03_tcp_ar_a/aaaaartn_s007_t019.edf"
#file = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaartn/s007_2014_11_09/03_tcp_ar_a/aaaaartn_s007_t018.edf"
file = Path.machine_path + "train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf"

def apply_montage(data):
    """Apply the bipolar montage"""
    channel_renaming_dict = {name: remove_suffix(name, ['-LE', '-REF']) for name in data.ch_names}
    data.rename_channels(channel_renaming_dict)
    #print(data.ch_names)
    bipolar_data = mne.set_bipolar_reference(
        data.load_data(), 
        anode=['FP1', 'F7', 'T3', 'T5',
             'FP1', 'F3', 'C3', 'P3',
             'FP2', 'F4', 'C4', 'P4', 
            'FP2', 'F8', 'T4', 'T6',
            'T3', 'C3', 'CZ', 'C4'], 
        cathode=['F7','T3', 'T5', 'O1', 
                 'F3', 'C3', 'P3', 'O1', 
                 'F4', 'C4', 'P4', 'O2', 
                 'F8', 'T4', 'T6', 'O2',
                 'C3', 'CZ', 'C4', 'T4'], 
        drop_refs=True)
    bipolar_data.pick_channels(['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
                            'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                            'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])
    return bipolar_data

def write_annotations(file, bipolar_data):
    """Read annotations from csv file and write them to the mne object"""
    df = pd.read_csv(file.removesuffix('edf') + 'csv_bi', header=5)
    onset_times = df['start_time'].values
    durations = df['stop_time'].values - df['start_time'].values 
    description = ["seizure"]
    annotations = mne.Annotations(onset_times, durations, description)
    bipolar_data.set_annotations(annotations)
    return bipolar_data

def plot_spectrogram_plt(bipolar_data):
    """Plot the spectrogram using matplotlib"""
    data_plot = bipolar_data.get_data()[0]
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data_plot, Fs=256)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    plt.subplot(2,1,1)
    plt.specgram(data_plot[650:], 256)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(2,1,2)
    plt.specgram(data_plot[540:640], 256)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_example_spectrogram():
    """This is an example of how to make a spectrogram"""
    frequencies = np.arange(5,105,5)
    samplingFrequency   = 400
    s1 = np.empty([0]) # For samples
    s2 = np.empty([0]) # For signal
    start = 1
    stop = samplingFrequency+1
    for frequency in frequencies:
        sub1 = np.arange(start, stop, 1)
    # Signal - Sine wave with varying frequency + Noise
        sub2 = np.sin(2*np.pi*sub1*frequency*1/samplingFrequency)+np.random.randn(len(sub1))
        s1 = np.append(s1, sub1)
        s2 = np.append(s2, sub2)
        start = stop+1
        stop = start+samplingFrequency

    # Plot the signal
    plt.subplot(211)
    plt.plot(s1,s2)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Plot the spectrogram
    plt.subplot(212)
    powerSpectrum2, freqenciesFound2, time2, imageAxis2 = plt.specgram(s2, Fs=samplingFrequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()   

def main():
    data = mne.io.read_raw_edf(file, infer_types=True)
    duration = data._raw_extras[0]['n_records']
    bipolar_data = apply_montage(data)
    bipolar_data = write_annotations(file, bipolar_data)
    epochs = mne.make_fixed_length_epochs(bipolar_data, duration=1)
    #check if the epoching worked by comparing the plots
    epochs.plot(n_epochs=5)
    bipolar_data.plot(duration=5,highpass=1, lowpass=70, n_channels=20)

    plot_spectrogram_plt(bipolar_data)
    plot_example_spectrogram()

if __name__ == "__main__":
    main()



"""
Load a pre-trained model and make a prediction on a signal. 
"""

import mne
import torch
import pandas as pd
from p_tools import load_data
from models import Model_ext_256

torch.set_default_dtype(torch.float64)


def sum_durations(onset_times,idx):
    if idx == len(onset_times)-1:
        return onset_times 
    elif onset_times[idx][0] + onset_times[idx][1] == onset_times[idx+1][0]:
        onset_times[idx][1] += 1
        onset_times.remove(onset_times[idx+1])
        sum_durations(onset_times, idx)
    else:
        sum_durations(onset_times, idx+1)

def combine_windows(onset_and_durations, allowed_overlap, idx):
    if idx == len(onset_and_durations)-1:
        return onset_and_durations
    elif onset_and_durations[idx][0] + onset_and_durations[idx][1] + allowed_overlap == onset_and_durations[idx+1][0]:  #allow for overlap
        onset_and_durations[idx][1] += onset_and_durations[idx+1][1] + allowed_overlap
        onset_and_durations.remove(onset_and_durations[idx+1])
        combine_windows(onset_and_durations, allowed_overlap, idx)
    else:
        combine_windows(onset_and_durations, allowed_overlap, idx+1)

def make_prediction(bipolar_data, epoch_tensor, model, threshold_prob, sum_durs=False, threshold_duration=0, combine_wins=False, allowed_overlap=0):
    predictions = model(epoch_tensor) #make predictions with model
    onset_times = torch.where(predictions.squeeze() > threshold_prob)[0].tolist() #indexes of epochs clasified as seizure. p > threshold_prob

    onsets_and_durations = [] #array to store onset times and durations
    for time in onset_times: 
        onsets_and_durations.append([time,1]) #every seizure epoch has duration 1
    if sum_durs:
        sum_durations(onsets_and_durations, idx=0) #concatenate consecutive seizure epochs and store new durations
    threshold_onset_and_durations = [item for item in onsets_and_durations if item[1] > threshold_duration] #remove seizures of duration < threshold_duration
    if combine_wins:
        combine_windows(threshold_onset_and_durations, allowed_overlap, idx=0)
    
    annotation_onsets = [onset for [onset,duration] in threshold_onset_and_durations]
    annotation_durations = [duration for [onset, duration] in threshold_onset_and_durations]

    annotations = mne.Annotations(annotation_onsets, annotation_durations, description="seizure") 

    bipolar_data.set_annotations(annotations)
    bipolar_data.plot(highpass=1, lowpass=70)


signal1 = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaaizz/s005_2010_10_12/03_tcp_ar_a/aaaaaizz_s005_t000.edf"
bipolar_data1, annotated_data1, epoch_tensor1, labels1 = load_data(signal1, epoch_length=1)
annotated_data1.plot(highpass=1, lowpass=70)


signal2 = "/home/migo/TUHP/TUSZ_V2/edf/eval/aaaaaaaq/s006_2014_08_18/01_tcp_ar/aaaaaaaq_s006_t000.edf"
bipolar_data2, annotated_data2, epoch_tensor2, labels2 = load_data(signal2, epoch_length=1)
annotated_data2.plot(highpass=1, lowpass=70)

model = Model_ext_256()
model.load_state_dict(torch.load("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/models/CNN_model.pt"))
model.eval()



make_prediction(bipolar_data1, epoch_tensor1, model=model, threshold_prob=0.5)
make_prediction(bipolar_data1, epoch_tensor1, model=model, threshold_prob=0.8)
make_prediction(bipolar_data1, epoch_tensor1, model=model, threshold_prob=0.5, sum_durs=True, threshold_duration=5)
make_prediction(bipolar_data1, epoch_tensor1, model=model, threshold_prob=0.5, sum_durs=True, threshold_duration=5, combine_wins=True, allowed_overlap=1)


make_prediction(bipolar_data2, epoch_tensor2, model=model, threshold_prob=0.5)
make_prediction(bipolar_data2, epoch_tensor2, model=model, threshold_prob=0.8)
make_prediction(bipolar_data2, epoch_tensor2, model=model, threshold_prob=0.5, sum_durs=True, threshold_duration=5)
make_prediction(bipolar_data2, epoch_tensor2, model=model, threshold_prob=0.5, sum_durs=True, threshold_duration=5, combine_wins=True, allowed_overlap=1)
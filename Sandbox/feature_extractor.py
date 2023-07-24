"""
Feature extraction script

Time-domain features:
1. Mean
2. Max (highest value)
3. Peak (largest absolute value)
4. Peak to peak (largest value - smallest value)
5. RMS
6. Variance
7. Standard Deviation
8. Power
9. Crest Factor  (Peak/RMS: ratio of peak values to effective value)
10. Form Factor (RMS / Mean(abs)) distortion of the waveform
11. Pulse Indicator (Peak / Mean?)
12. Margin (Peak / abs(Power)^2)
13. Kurtosis (standardized 4th moment of distribution: measurement of heaviness of tails relative to normal distribution)
14. Skewness (assymetry of distribution)

Frequency-domain features (requires FFT and corresponding power spectrum):
15. Mean of band Power Spectrum
16. Max of band Power Spectrim 
17. Sum of total band power
18. Peak of band power
19. Variance of band power
20. Standard Deviation of band power
21. Skewness of band power
22. Kurtosis of band power
23. Relative Spectral Peak per band

"""

import torch
import glob
import mne
from p_tools import apply_montage

tensor = torch.load("/Users/toucanfirm/Documents/DTU/Speciale/tools/data/aaaaaacz/s006_2015_10_05/03_tcp_ar_a/aaaaaacz_s006_t000/60.pt")
time_features = torch.tensor([20,14])

mean = tensor.mean(1)
max = tensor.max(1)[0]
peak = tensor.abs().max(1)[0]
rms = ((tensor.square().sum(1))/tensor.shape[1]).sqrt()
std = tensor.std(1)
power = tensor.square().sum(1) / tensor.shape[1]
crest = peak/rms
form = rms / tensor.abs().mean(1)
pulse = peak / tensor.abs().mean(1)
margin = 


# for f in glob.glob("/Users/toucanfirm/Documents/DTU/Speciale/tools/data/aaaaaacz/s006_2015_10_05/03_tcp_ar_a/aaaaaacz_s006_t000" + '/**/*.pt', recursive=True):
#     tensor = torch.load(f)
#     sum = tensor.sum()
#     if sum == 0:
#         zero_epochs.append(f)

# for i in range(len(zero_epochs)):
#     epoch = zero_epochs[i].split('/')[-1]
#     print(epoch)


# data = mne.io.read_raw_edf("/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaaacz/s006_2015_10_05/03_tcp_ar_a/aaaaaacz_s006_t000.edf", infer_types=True)
# bipolar = apply_montage(data)
# raw = data.get_data()
# bi_raw = bipolar.get_data()

# data.plot()
# bipolar.plot()

    # channel_renaming_dict = {name: remove_suffix(name, ['-LE', '-REF']) for name in data.ch_names}
    # data.rename_channels(channel_renaming_dict)
    # #print(data.ch_names)
    # bipolar_data = mne.set_bipolar_reference(
    #     data.load_data(), 
    #     anode=['FP1', 'F7', 'T3', 'T5',
    #          'FP1', 'F3', 'C3', 'P3',
    #          'FP2', 'F4', 'C4', 'P4', 
    #         'FP2', 'F8', 'T4', 'T6',
    #         'T3', 'C3', 'CZ', 'C4'], 
    #     cathode=['F7','T3', 'T5', 'O1', 
    #              'F3', 'C3', 'P3', 'O1', 
    #              'F4', 'C4', 'P4', 'O2', 
    #              'F8', 'T4', 'T6', 'O2',
    #              'C3', 'CZ', 'C4', 'T4'], 
    #     drop_refs=True)
    # bipolar_data.pick_channels(['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
    #                         'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    #                         'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
    #                         'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
    #                         'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4'])
    # return bipolar_data
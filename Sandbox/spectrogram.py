import mne 
import p_tools
import matplotlib.pyplot as plt

file1 = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaartn/s007_2014_11_09/03_tcp_ar_a/aaaaartn_s007_t003.edf"
file2 = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaakmu/s001_2010_11_01/03_tcp_ar_a/aaaaakmu_s001_t000.edf"
file3 = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaaiat/s008_2012_09_11/01_tcp_ar/aaaaaiat_s008_t000.edf"
file4 = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaanrc/s004_2012_10_11/01_tcp_ar/aaaaanrc_s004_t010.edf"

bipolar_data1, annotated_data1, epoch_tensor1, labels1 = p_tools.load_data(file1, epoch_length=1)
bipolar_data2, annotated_data2, epoch_tensor2, labels2 = p_tools.load_data(file2, epoch_length=1)
bipolar_data3, annotated_data3, epoch_tensor3, labels3 = p_tools.load_data(file3, epoch_length=1)
bipolar_data4, annotated_data4, epoch_tensor4, labels4 = p_tools.load_data(file4, epoch_length=1)



plt.figure(1)
plt.subplot(221)
plt.specgram(bipolar_data1.get_data()[0], Fs=bipolar_data1.info['sfreq'])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("aaaaartn_s007_t003")
plt.subplot(222)
plt.specgram(bipolar_data2.get_data()[0], Fs=bipolar_data2.info['sfreq'])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("aaaaakmu_s001_t000")
plt.subplot(223)
plt.specgram(bipolar_data3.get_data()[0], Fs=bipolar_data3.info['sfreq'])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("aaaaaiat_s008_t000")
plt.subplot(224)
plt.specgram(bipolar_data4.get_data()[0], Fs=bipolar_data4.info['sfreq'])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("aaaaanrc_s004_t010")
plt.show()

plt.figure(2)
plt.subplot(221)
plt.specgram(bipolar_data1.notch_filter(60).get_data()[0], Fs=bipolar_data1.info['sfreq'])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("aaaaartn_s007_t003")
plt.subplot(222)
plt.specgram(bipolar_data2.notch_filter(60).get_data()[0], Fs=bipolar_data2.info['sfreq'])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("aaaaakmu_s001_t000")
plt.subplot(223)
plt.specgram(bipolar_data3.notch_filter(60).get_data()[0], Fs=bipolar_data3.info['sfreq'])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("aaaaaiat_s008_t000")
plt.subplot(224)
plt.specgram(bipolar_data4.notch_filter(60).get_data()[0], Fs=bipolar_data4.info['sfreq'])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("aaaaanrc_s004_t010")
plt.show()



# p_tools.plot_spectrogram_plt(bipolar_data1)
# p_tools.plot_spectrogram_plt(bipolar_data2)
# p_tools.plot_spectrogram_plt(bipolar_data3)
# p_tools.plot_spectrogram_plt(bipolar_data4)



#     data_plot = bipolar_data.get_data()[0]
#     powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data_plot, Fs=256)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#     plt.show()
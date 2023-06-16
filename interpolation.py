import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import mne
import p_tools


file256 = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaapwy/s007_2014_06_17/01_tcp_ar/aaaaapwy_s007_t002.edf"
file400 = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaabom/s005_2006_01_23/03_tcp_ar_a/aaaaabom_s005_t007.edf"
file512 = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaalep/s008_2012_04_26/01_tcp_ar/aaaaalep_s008_t008.edf"
file1000 = "/home/migo/TUHP/TUSZ_V2/edf/train/aaaaampz/s002_2015_01_23/01_tcp_ar/aaaaampz_s002_t000.edf"

bipolar_data256, annotated_data256, epoch_tensor256, labels256 = p_tools.load_data(file256, epoch_length=1)
bipolar_data400, annotated_data400, epoch_tensor400, labels400 = p_tools.load_data(file400, epoch_length=1)
bipolar_data512, annotated_data512, epoch_tensor512, labels512 = p_tools.load_data(file512, epoch_length=1)
bipolar_data1000, annotated_data1000, epoch_tensor1000, labels1000 = p_tools.load_data(file1000, epoch_length=1)
#bipolar_data.plot(duration=1, n_channels=1)

signal256 = epoch_tensor256[0][0].numpy()
signal400 = epoch_tensor400[0][0].numpy()
signal512 = epoch_tensor512[0][0].numpy()
signal1000 = epoch_tensor1000[0][0].numpy()

# plt.plot(signal1)
# plt.xlabel("Index")
# plt.ylabel("Fp1-F7 [microvolts]")
# plt.show()

x_250 = np.linspace(0,1,250)
x_256 = np.linspace(0,1,256)
x_400 = np.linspace(0,1,400)
x_512 = np.linspace(0,1,512)
x_1000 = np.linspace(0,1,1000)


plt.figure(1)
plt.subplot(221)
plt.plot(x_256, signal256)
plt.subplot(222)
plt.plot(x_400, signal400)
plt.subplot(223)
plt.plot(x_512, signal512)
plt.subplot(224)
plt.plot(x_1000, signal1000)

resampled256 = signal.resample(signal256, 250)
resampled_poly256 = signal.resample_poly(signal256, 125, 128)
resampled400 = signal.resample(signal400, 250) 
resampled_poly400 = signal.resample_poly(signal400, 5, 8)
resampled512 = signal.resample(signal512, 250)
resampled_poly512 = signal.resample_poly(signal512, 125, 256)
resampled1000 = signal.resample(signal1000, 250) 
resampled_poly1000 = signal.resample_poly(signal1000, 1, 4)

plt.figure(2)
plt.title("Different frequencies resampled at 250Hz using two methods")
plt.subplot(421)
plt.plot(x_256, signal256, x_250, resampled256, 'g-')
plt.title("256 Hz")
plt.legend(['data', 'resampled'])
plt.subplot(422)
plt.plot(x_256, signal256, x_250, resampled_poly256)
plt.title("256 Hz")
plt.legend(['data', 'resampled_poly'])

plt.subplot(423)
plt.plot(x_400, signal400, x_250, resampled400, 'g.-')
plt.title("400 Hz")
plt.legend(['data', 'resampled'])
plt.subplot(424)
plt.plot(x_400, signal400, x_250, resampled_poly400)
plt.title("400 Hz")
plt.legend(['data', 'resampled_poly'])

plt.subplot(425)
plt.plot(x_512, signal512, x_250, resampled512, 'g.-')
plt.title("512 Hz")
plt.legend(['data', 'resampled'])
plt.subplot(426)
plt.plot(x_512, signal512, x_250, resampled_poly512)
plt.title("512 Hz")
plt.legend(['data', 'resampled_poly'])

plt.subplot(427)
plt.plot(x_1000, signal1000, x_250, resampled1000, 'g.-')
plt.legend(['data', 'resampled'])
plt.title("1000 Hz")
plt.subplot(428)
plt.plot(x_1000, signal1000, x_250, resampled_poly1000)
plt.legend(['data', 'resampled_poly'])
plt.title("1000 Hz")
plt.show()


#now we filter the data 1Hz-70Hz before we resample

# filtered256 = mne.filter.filter_data(signal256, 256, 1, 70)
# filtered400 = mne.filter.filter_data(signal400, 400, 1, 70)
# filtered512 = mne.filter.filter_data(signal512, 512, 1, 70)
# filtered1000 = mne.filter.filter_data(signal1000, 1000, 1, 70)


# plt.figure(3)
# plt.subplot(411)
# plt.plot(x_256, signal256, x_256, filtered256)
# plt.legend(['raw', 'filtered'])

# plt.subplot(412)
# plt.plot(x_400, signal400, x_400, filtered400)
# plt.legend(['raw', 'filtered'])

# plt.subplot(413)
# plt.plot(x_512, signal512, x_512, filtered512)
# plt.legend(['raw', 'filtered'])

# plt.subplot(414)
# plt.plot(x_1000, signal1000, x_1000, filtered1000)
# plt.legend(['raw', 'filtered'])
# plt.show()

# x = np.linspace(0, 10, 20, endpoint=False)
# y = np.cos(-x**2/6.0)
# f_fft = signal.resample(y, 100)
# f_poly = signal.resample_poly(y, 100, 20)
# xnew = np.linspace(0, 10, 100, endpoint=False)
# plt.plot(xnew, f_fft, 'b.-', xnew, f_poly, 'r.-')
# plt.plot(x, y, 'ko-')
# plt.plot(10, y[0], 'bo', 10, 0., 'ro')  # boundaries
# plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
# plt.show()
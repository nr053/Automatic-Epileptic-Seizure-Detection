import numpy as np
from scipy.fft import fft, rfft, irfft, ifft
from scipy.fft import fftfreq, rfftfreq
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import mne
import p_tools

x_space = np.linspace(0,2,400)

signal1 = 3 * np.sin(x_space * np.pi * 2)
signal2 = np.sin(2 * np.pi * x_space * 20)
signal3 = 0.5 * np.sin(2 * np.pi * x_space * 10)

signal = signal1 + signal2 + signal3


plt.figure(1)
plt.subplot(231)
plt.plot(x_space, signal1)
plt.title("Signal 1")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.subplot(232)
plt.plot(x_space, signal2)
plt.title("Signal 2")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.subplot(233)
plt.plot(x_space, signal3)
plt.title("Signal 3")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.subplot(235)
plt.plot(x_space, signal)
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.title("Signals")
plt.show()


fourier = fft(signal)


plt.figure(2)

plt.subplot(231)
plt.plot(x_space, signal)
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.title("Combined Signals")

plt.subplot(232)
plt.plot(np.abs(fourier))
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.title('FFT Spectrum')


N = len(signal)
normalise = N/2

# Plot the normalized FFT (|Xk|)/(N/2)
plt.subplot(233)
plt.plot(np.abs(fourier)/normalise)
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.title('Normalized FFT Spectrum')



# Plot the normalized FFT (|Xk|)/(N/2) on the frequency axis
plt.subplot(234)
plt.plot(fftfreq(n=400, d=1/200), np.abs(fourier)/normalise)
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.title('Normalized FFT Spectrum')




# Plot the actual spectrum of the signal
plt.subplot(235)
plt.plot(rfftfreq(n=400, d=1/200), 2*np.abs(rfft(signal))/N)
plt.title('Normalized FFT Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')



rebuilt = ifft(fourier)

plt.subplot(236)
plt.plot(rebuilt)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.title("Rebuilt signal")
plt.show()


data = mne.io.read_raw_edf("/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaajqo/s001_2010_04_27/01_tcp_ar/aaaaajqo_s001_t000.edf", infer_types=True)
bipolar = p_tools.apply_montage(data)
signal_values = bipolar.get_data()

plt.figure(3)
plt.subplot(231)
plt.plot(signal_values[0][20*250:(20*250 + 250)])
plt.title("EEG")
plt.xlabel('Samples')
plt.ylabel('Amplitude')

EEG = signal_values[0][20*250:(20*250 + 250)]

fourier2 = fft(EEG)


plt.subplot(232)
plt.plot(np.abs(fourier2))
plt.xlabel('Samples')
plt.ylabel('Frequency [Hz]')
plt.title('FFT Spectrum')

N = len(EEG)
normalise=N/2

# Plot the normalized FFT (|Xk|)/(N/2)
plt.subplot(233)
plt.plot(np.abs(fourier2)/normalise)
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.title('Normalized FFT Spectrum')



# Plot the normalized FFT (|Xk|)/(N/2) on the frequency axis
plt.subplot(234)
plt.plot(fftfreq(n=250, d=1/250), np.abs(fourier2)/normalise)
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.title('Normalized FFT Spectrum')



# Plot the actual spectrum of the signal
plt.subplot(235)
plt.plot(rfftfreq(n=250, d=1/250), 2*np.abs(rfft(EEG))/N)
plt.title('Spectrum')
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')




rebuilt2 = ifft(fourier2)

plt.subplot(236)
plt.plot(rebuilt2)
plt.title("rebuilt signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()





#same again but using the real Fourier Transform

plt.figure(4)
plt.subplot(231)
plt.plot(signal_values[0][20*250:(20*250 + 250)])
plt.title("EEG")
plt.xlabel('Samples')
plt.ylabel('Amplitude')

EEG = signal_values[0][20*250:(20*250 + 250)]

fourier2 = rfft(EEG)


plt.subplot(232)
plt.plot(np.abs(fourier2))
plt.xlabel('Samples')
plt.ylabel('Frequency [Hz]')
plt.title('FFT Spectrum')

N = len(EEG)
normalise=N/2

# Plot the normalized FFT (|Xk|)/(N/2)
plt.subplot(233)
plt.plot(np.abs(fourier2)/normalise)
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.title('Normalized FFT Spectrum')



# Plot the normalized FFT (|Xk|)/(N/2) on the frequency axis
plt.subplot(234)
plt.plot(rfftfreq(n=250, d=1/250), np.abs(fourier2)/normalise)
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.title('Normalized FFT Spectrum')



# Plot the actual spectrum of the signal
plt.subplot(235)
plt.plot(rfftfreq(n=250, d=1/250), 2*np.abs(rfft(EEG))/N)
plt.title('Spectrum')
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')




rebuilt2 = irfft(fourier2)

plt.subplot(236)
plt.plot(rebuilt2)
plt.title("rebuilt signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()


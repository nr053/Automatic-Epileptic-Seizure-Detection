import mne 
import p_tools


file1 = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaartn/s007_2014_11_09/03_tcp_ar_a/aaaaartn_s007_t003.edf"
file2 = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaakmu/s001_2010_11_01/03_tcp_ar_a/aaaaakmu_s001_t000.edf"
file3 = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaaiat/s008_2012_09_11/01_tcp_ar/aaaaaiat_s008_t000.edf"
file4 = "/Users/toucanfirm/Documents/DTU/Speciale/TUSZ_V2/edf/train/aaaaanrc/s004_2012_10_11/01_tcp_ar/aaaaanrc_s004_t010.edf"

bipolar_data1, annotated_data1, epoch_tensor1, labels1 = p_tools.load_data(file1, epoch_length=1)
bipolar_data2, annotated_data2, epoch_tensor2, labels2 = p_tools.load_data(file2, epoch_length=1)
bipolar_data3, annotated_data3, epoch_tensor3, labels3 = p_tools.load_data(file3, epoch_length=1)
bipolar_data4, annotated_data4, epoch_tensor4, labels4 = p_tools.load_data(file4, epoch_length=1)

p_tools.plot_spectrogram_plt(bipolar_data1)
p_tools.plot_spectrogram_plt(bipolar_data2)
p_tools.plot_spectrogram_plt(bipolar_data3)
p_tools.plot_spectrogram_plt(bipolar_data4)
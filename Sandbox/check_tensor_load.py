import mne 
import torch
from p_tools import load_data, apply_montage, make_labels
from random import sample

epoch_length = 1

with open('file_list_256.txt') as f:
    lines = [line.rstrip() for line in f]



sample_list = sample(lines, 10)
print(f"Sample of files: {sample_list}")

for i in range(len(sample_list)):
    edf_object, label_df = load_data(sample_list[i])
    bipolar_object = apply_montage(edf_object)
    epochs = mne.make_fixed_length_epochs(bipolar_object, duration=epoch_length)
    epoch_tensor = torch.tensor(epochs.get_data())
    labels = make_labels(epoch_tensor, label_df)

    random_epochs = sample(range(0, len(epoch_tensor)), 10)

    for epoch in random_epochs:
        tmp_tensor = torch.load('data/' + '/'.join(sample_list[i].split('/')[9:]).removesuffix('.edf') + '/' + str(epoch) + '.pt')

        print(f"Random epoch: {epoch}")
        print(epoch_tensor[epoch].shape)
        print(tmp_tensor.shape)
        print(torch.equal(epoch_tensor[epoch], tmp_tensor))



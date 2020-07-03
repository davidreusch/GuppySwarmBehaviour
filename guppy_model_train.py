import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from view_hdf import get_locomotion_vec, Guppy_Calculator, Guppy_Dataset
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from guppy_model import LSTM_fixed, LSTM_multi_modal
import sys
import copy
from hyper_params import *

torch.manual_seed(1)

# get the files for 4, 6 and 8 guppys
trainpath = "guppy_data/live_female_female/train/" if live_data else "guppy_data/couzin_torus/train/"
files = [join(trainpath, f) for f in listdir(trainpath) if isfile(join(trainpath, f)) and f.endswith(".hdf5")]
files.sort()
num_files = len(files) - 6
files = files[num_files:]
print(files)

torch.set_default_dtype(torch.float64)

# now we use a regression model, just predict the absolute values of linear speed and angular turn
# so we need squared_error loss

if output_model == "multi_modal":
    model = LSTM_multi_modal()
    loss_function = nn.CrossEntropyLoss()
else:
    model = LSTM_fixed()
    loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
print(model)
# training

dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=live_data, output_model=output_model)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

epochs = 15
for i in range(epochs):
    h = model.init_hidden(batch_size, num_layers)
    for inputs, targets in dataloader:
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        model.zero_grad()
        h = tuple([each.data for each in h])

        if output_model == "multi_modal":
            targets = targets.type(torch.LongTensor)
            angle_pred, speed_pred, h = model.forward(inputs, h)
            angle_pred = angle_pred.view(angle_pred.shape[0] * angle_pred.shape[1], -1)
            speed_pred = speed_pred.view(speed_pred.shape[0] * speed_pred.shape[1], -1)
            targets = targets.view(targets.shape[0] * targets.shape[1], 2)
            angle_targets = targets[:, 0]
            speed_targets = targets[:, 1]

            loss1 = loss_function(angle_pred, angle_targets)
            loss2 = loss_function(speed_pred, speed_targets)
            loss = loss1 + loss2

        else:
            prediction, h = model.forward(inputs, h)
            loss = loss_function(prediction, targets)

        loss.backward()
        optimizer.step()

    print(f'epoch: {i:3} loss: {loss.item():10.10f}')

network_path = "guppy_net_live.pth" if live_data else "guppy_net.pth"
torch.save(model.state_dict(), network_path)


#
#
# num_steps = 25
# # from https://github.com/LeanManager/NLP-PyTorch/blob/master/Character-Level%20LSTM%20with%20PyTorch.ipynb
# def get_batches(files, num_files, n_steps):
#     '''
#     Create a list of batches of size
#        num_files x n_steps x inputsize
#        from the files
#
#        num_files is our batch size, the number of sequences per batch
#        n_steps is the length of each sequence in a batch
#        inputsize is just the length of our handcrafted vector
#
#     '''
#
#     #get the data and append it to the array
#     arr = []
#     for i in range(num_files):
#         gc = Guppy_Calculator(files[i], agent, num_guppy_bins, num_wall_rays, livedata=False)
#         data = gc.get_data_old()
#         # print(data.shape)
#         arr.append(data)
#
#     # array has size (num_files, len(track)=750, input_dim)
#     arr = np.array(arr)
#
#     batch_size = num_files * n_steps
#     n_batches = len(arr) // batch_size
#
#     # Keep only enough characters to make full batches
#     # arr = arr[:n_batches * batch_size]
#
#     res = []
#     for n in range(0, arr.shape[1], n_steps):
#
#         # The features
#         x = arr[:, n:n + n_steps, :]
#         #x = arr[:, : n_steps, :]
#
#         # The targets, shifted by one
#         y = np.zeros_like(x)
#
#         # shift the targets
#         try:
#             y[:, :-1, :], y[:, -1, :] = x[:, 1:, :], arr[:, n + n_steps, :]
#             #y[:, :-1, :], y[:, -1, :] = x[:, 1:, :], arr[:,  n_steps, :]
#         except IndexError:
#             y[:, :-1, :], y[:, -1, :] = x[:, 1:, :], arr[:, n + n_steps - 2, :]
#             #y[:, :-1, :], y[:, -1, :] = x[:, 1:, :], arr[:, n_steps - 1, :]
#
#         res.append((torch.from_numpy(x), torch.from_numpy(np.apply_along_axis(one_hot_wrap, 2, y))))
#
#     return res

# """
# data = get_batches(files, num_files, num_steps)
# for i in range(epochs):
#     h = model.init_hidden(num_files, num_layers)
#     counter = 0
#     for x, y in data:
#         inputs, targets = x, y
#
#         # Creating new variables for the hidden state, otherwise
#         # we'd backprop through the entire training history
#         h = tuple([each.data for each in h])
#
#         model.zero_grad()
#
#         angle_out, speed_out, h = model.forward(inputs, h)
#
#         #unroll the sequences and the targets for the loss function
#         angle_out = angle_out.view(num_files * num_steps, angle_bins)
#         speed_out = speed_out.view(num_files * num_steps, speed_bins)
#         targets = targets.view(num_files * num_steps, 2)
#
#         angle_targets = targets[:, 0]
#         speed_targets = targets[:, 1]
#
#         m = nn.Softmax(dim=1)
#         angle_pred = m(angle_out)
#         speed_pred = m(speed_out)
#         print("angle_output: ")
#         print(angle_out.size())
#         print(angle_out)
#         print("angle_prediction: ")
#         print(angle_pred.size())
#         print(angle_pred)
#         print("speed_output: ")
#         print(speed_out.size())
#         print(speed_out)
#         print("speed_prediction: ")
#         print(speed_pred.size())
#         print(speed_pred)
#
#         # cross entropy should work without the softmax as far as I understand
#         loss1 = loss_function(angle_out, angle_targets)
#         loss2 = loss_function(speed_out, speed_targets)
#         loss = loss1 + loss2
#         loss.backward(retain_graph=True)
#         optimizer.step()
#
#         #print every 250th time
#         if counter % 250 == 0:
#             print(f'epoch: {i:3} loss: {loss.item():10.10f}')
#         counter += 1
#
#     print(f'epoch: {i:3} loss: {loss.item():10.10f}')
#
# """
# old code
"""
for i in range(epochs):
    for seq, labels in train_data:
        optimizer.zero_grad()
        model.hidden_cell = (torch.randn(1, 1, model.hidden_layer_size),
                             torch.randn(1, 1, model.hidden_layer_size))

        y_pred = model.forward(seq)
        # print(seq)
        # print("Prediction: ", y_pred)
        # print("Labels ", labels)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()  # retain_graph=True)
        optimizer.step()
        if i % 1000 == 0:
            print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# prediction

# fut_pred = len(train_data)
fut_pred = len(test_data)
train_pred = []
test_pred = []

model.eval()

for i in range(fut_pred):
    # seq = torch.FloatTensor(train_data[i])
    seq = torch.FloatTensor(test_data[i])
    # print(model(seq))
    # print(test_data[i])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_pred.append(model(seq).numpy())  # .item()
        # train_pred.append(model(seq).numpy()) #.item()

test_pred = torch.FloatTensor(test_pred)
test_pred = test_pred.reshape(fut_pred, 1, 4)
# train_pred=torch.FloatTensor(train_pred)
# train_pred=train_pred.reshape(fut_pred,1,4)

# plt.scatter(train_data[:,:,0],train_data[:,:,1])
# plt.scatter(train_data[0,:,0],train_data[0,:,1])
# plt.plot(train_pred[:,:,0],train_pred[:,:,1])

plt.scatter(test_data[:, :, 0], test_data[:, :, 1])
plt.scatter(test_data[0, :, 0], test_data[0, :, 1])
plt.scatter(test_pred[:, :, 0], test_pred[:, :, 1])
plt.scatter(test_pred[0, :, 0], test_pred[0, :, 1])

plt.show()

# old function to create the data
seq_len = 700


def createDataSet(files):
    inout_seq = []
    for filename in files:
        gc = Guppy_Calculator(filename, agent, num_guppy_bins, num_wall_rays, livedata=False)
        next = gc.craft_vector(0)
        # for i in range(1, gc.length):
        for i in range(gc.length - seq_len):
            seq = []
            for j in range(i, seq_len + i):
                # train_seq = torch.tensor(next)
                # next = gc.craft_vector(j)
                seq.append(torch.tensor(gc.craft_vector(j)))

            label_data = gc.craft_vector(j + 1)
            # print(label_data)
            angle_label = one_hot(label_data[0], -0.04, 0.04, angle_bins, get_class=True)
            speed_label = one_hot(label_data[1], 0.0, 0.4, speed_bins, get_class=True)
            label = torch.tensor([angle_label, speed_label])
            # print(label)
            # angle_label = one_hot(next[0], -0.04, 0.04, disc_angle_bins, get_class=True)
            # speed_label = one_hot(next[1], 0.0, 0.4, disc_speed_bins, get_class=True)
            seq = torch.cat(seq).view(len(seq), 1, -1)
            inout_seq.append((seq, label))
    return inout_seq

# train_data = createDataSet(files[0:1])

# print(train_data[0][0].size())


# for (i,j) in train_data:
# print("data: ", i)
# print("label: ", j)

# test_inout_seq = create_inout_sequences(test_data, train_window)
"""

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
files = [join(trainpath, f) for f in listdir(trainpath) if isfile(join(trainpath, f)) and f.endswith(".hdf5") ]
files.sort()
num_files = len(files) // 2
files = files[-20:]
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)
# training

dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=live_data, output_model=output_model, max_agents= 1)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

epochs = 30
seq_len = 20
for i in range(epochs):
    try:
        #states = [model.init_hidden(batch_size, 1, hidden_layer_size) for _ in range(num_layers * 2)]
        #h = model.init_hidden(batch_size, num_layers, hidden_layer_size)
        #loss = 0
        for inputs, targets in dataloader:
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            optimizer.zero_grad()
            #h = tuple([each.data for each in h])
            #states = [tuple([each.data for each in s]) for s in states]

            if output_model == "multi_modal":
                targets = targets.type(torch.LongTensor)
                angle_pred, speed_pred, h = model.forward(inputs, h)
                #angle_pred, speed_pred, states = model.forward(inputs, states)
                #print(angle_pred.size())
                #print(speed_pred.size())

                #print(targets.size())
                angle_pred = angle_pred.view(angle_pred.shape[0] * angle_pred.shape[1], -1)
                speed_pred = speed_pred.view(speed_pred.shape[0] * speed_pred.shape[1], -1)
                #print(angle_pred.size())
                #print(speed_pred.size())


                # #angle_speed, states = model.forward(inputs, states)
                # angle_speed = angle_speed.view(batch_size * angle_speed.size()[1], -1)
                # angle_bins = angle_speed[:, :num_angle_bins]
                # speed_bins = angle_speed[:, num_angle_bins:]
                # #angle_bins = angle_speed.view(-1, num_angle_bins)
                # #speed_bins = angle_speed.view(-1, num_speed_bins)
                # print(angle_bins.size())
                # print(speed_bins.size())
                #
                targets = targets.view(targets.shape[0] * targets.shape[1], 2)
                # print(targets.size())
                angle_targets = targets[:, 0]

                speed_targets = targets[:, 1]
                #print(speed_targets)

                loss1 = loss_function(angle_pred, angle_targets)
                loss2 = loss_function(speed_pred, speed_targets)
                loss = loss1 + loss2

                torch.set_printoptions(threshold=10000)
                with torch.no_grad():
                     for j in range(5):
                    #     print(nn.Softmax(0)(speed_pred[0]))
                    #     print("------SPEED TARGETS -------")
                    #     print(speed_targets)
                         angle_probs = nn.Softmax(0)(angle_pred[i])
                         speed_probs = nn.Softmax(0)(speed_pred[i])
                         print("angle prob:\n", angle_probs[angle_targets[i].data])
                         print(angle_targets[i])
                         #print("speed prob:\n", speed_probs[speed_targets[i].data])
                         #print(speed_targets[i])



            else:
                h = model.init_hidden(batch_size, num_layers, hidden_layer_size)
                for s in range(0, inputs.size()[1] - seq_len, seq_len):
                    #prediction, states = model.forward(inputs[:, s:seq_len, :], states)
                    prediction, h = model.forward(inputs[:, s:s + seq_len, :], h)
                    loss = loss_function(prediction, targets[:, s: s + seq_len, :])


            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            loss.backward()
            optimizer.step()

    except KeyboardInterrupt:
        if input("Do you want to save the model trained so far? y/n") == "y":
            torch.save(model.state_dict(), network_path + f".epochs{i}")
            print("network saved at " + network_path + f".epochs{i}")
        sys.exit(0)

    print(f'epoch: {i:3} loss: {loss.item():10.10f}')

torch.save(model.state_dict(), network_path + f".epochs{epochs}")
print("network saved at " + network_path + f".epochs{epochs}")



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
from auxiliary_funcs import plot_scores
torch.manual_seed(1)
torch.set_default_dtype(torch.float64)

# get the files for 4, 6 and 8 guppys
trainpath = "guppy_data/live_female_female/train/" if live_data else "guppy_data/couzin_torus/train/"
files = [join(trainpath, f) for f in listdir(trainpath) if isfile(join(trainpath, f)) and f.endswith(".hdf5") ]
files.sort()
num_files = len(files) // 2
files = files[:]
print(len(files))
dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=live_data, output_model=output_model, max_agents= 1)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
print(dataset.filepaths)

valpath = "guppy_data/live_female_female/validation/" if live_data else "guppy_data/couzin_torus/validation/"
valfiles = [join(valpath, f) for f in listdir(valpath) if isfile(join(valpath, f)) and f.endswith(".hdf5") ]
valfiles.sort()
valset = Guppy_Dataset(valfiles, 0, num_guppy_bins, num_wall_rays, livedata=live_data, output_model=output_model, max_agents= 1)
valloader = DataLoader(valset, batch_size=valbatch_size, drop_last=True, shuffle=True)
print(valset.filepaths)

# now we use a regression model, just predict the absolute values of linear speed and angular turn
# so we need squared_error loss

models = {"fixed": LSTM_fixed(arch=""),
          "fixedey": LSTM_fixed(arch="ey"),
          "multi_modal": LSTM_multi_modal(arch=""),
          "multi_modaley": LSTM_multi_modal(arch="ey")}

model = models[hyperparams["overall_model"]]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = model.lossfn
print(model)


# training
epochs = 15
train_losses_per_epoch = []
val_losses_per_epoch = []
for i in range(epochs):
    try:
        #training
        model.train()
        average_loss = 0
        for inputs, targets in dataloader:
            # Creating new variables for the hidden state, otherwise we'd backprop through the entire training history
            #states = [tuple([each.data for each in s]) for s in states]
            optimizer.zero_grad()
            prediction, states, loss = model.do_batch(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            average_loss += loss.item()

        average_loss /= len(dataloader)
        train_losses_per_epoch.append(average_loss)
        print(f'################\nepoch: {i:3} loss: {average_loss:10.10f}\n###################')

        # validation
        with torch.no_grad():
            model.eval()
            val_losses = []
            val_loss = 0
            for inputs, targets in valloader:
                prediction, states, loss = model.do_batch(inputs, targets, batch_size=valbatch_size)
                val_loss += loss.item()

            val_loss /= len(valloader)
            val_losses_per_epoch.append(val_loss)
            print(f'################\nepoch: {i:3} valloss: {val_loss:10.10f}\n###################')

    except KeyboardInterrupt:
        if input("Do you want to save the model trained so far? y/n") == "n":
            sys.exit(0)
        break

torch.save(model.state_dict(), network_path + f".epochs{epochs}")
print("network saved at " + network_path + f".epochs{epochs}")
with open("saved_networks/last_trained_net.txt", "w") as f:
    f.write(network_path + f".epochs{epochs}")

scores = [train_losses_per_epoch, val_losses_per_epoch]
plot_scores(scores, len(scores[0]))



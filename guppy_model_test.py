import numpy as np
import torch
import torch.nn as nn
from guppy_model import *
from os.path import isfile, join
from os import listdir
from view_hdf import Guppy_Dataset
from torch.utils.data import Dataset, DataLoader
from hyper_params import *
torch.set_default_dtype(torch.float64)
model = LSTM_fixed()
model.load_state_dict(torch.load(network_path))
model.eval()

testpath = "guppy_data/live_female_female/test/" if live_data else "guppy_data/couzin_torus/test/"
files = [join(testpath, f) for f in listdir(testpath) if
         isfile(join(testpath, f)) and f.endswith(".hdf5")]
files.sort()
num_files = len(files)
files = files[:num_files]
print(files)

dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=False)
testloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

with torch.no_grad():
    for x, y in testloader:
        model.predict(x, y)




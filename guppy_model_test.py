import numpy as np
import torch
import torch.nn as nn
from guppy_model import *
from os.path import isfile, join
from os import listdir
from view_hdf import Guppy_Dataset
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float64)
PATH = "guppy_net.pth"
model = LSTM()
model.load_state_dict(torch.load(PATH))
model.eval()

batch_size = 4
mypath = "guppy_data/couzin_torus/test/"
files = [join(mypath, f) for f in listdir(mypath) if
         isfile(join(mypath, f))]
files.sort()
num_files = len(files)
files = files[:num_files]
print(files)

dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=False)
testloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

with torch.no_grad():
    for x, y in testloader:
        model.predict(x, y)




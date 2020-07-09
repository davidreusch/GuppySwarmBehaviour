import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from view_hdf import get_locomotion_vec, Guppy_Calculator, Guppy_Dataset, value_to_bin
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from guppy_model import LSTM_fixed, LSTM_multi_modal
import sys
import copy
from hyper_params import *


trainpath = "guppy_data/live_female_female/train/" if live_data else "guppy_data/couzin_torus/train/"
files = [join(trainpath, f) for f in listdir(trainpath) if isfile(join(trainpath, f)) and f.endswith(".hdf5")]
files.sort()
num_files = len(files)
files = files[:num_files]
print(files)


min_angle, max_angle, min_speed, max_speed = \
    Guppy_Calculator(files[0], agent, num_guppy_bins, num_wall_rays, livedata=live_data).get_min_max_angle_speed()
for filepath in files:
    gc = Guppy_Calculator(filepath, agent, num_guppy_bins, num_wall_rays, livedata=live_data)
    pot_min_angle, pot_max_angle, pot_min_speed, pot_max_speed = gc.get_min_max_angle_speed()
    min_angle = pot_min_angle if pot_min_angle < min_angle else min_angle
    max_angle = pot_max_angle if pot_max_angle > max_angle else max_angle
    min_speed = pot_min_speed if pot_min_speed < min_speed else min_speed
    max_speed = pot_max_speed if pot_max_speed > max_speed else max_speed
    print("Min_Angle:", pot_min_angle)
    print("Max_Angle:", pot_max_angle)
    print("Min_Speed:", pot_min_speed)
    print("Max_Speed:", pot_max_speed)

print("Min_Angle:", min_angle)
print("Max_Angle:", max_angle)
print("Min_Speed:", min_speed)
print("Max_Speed:", max_speed)



print(value_to_bin(-0.6, -0.4, 0.4, 10))
print(value_to_bin(0.1, -0.4, 0.4, 10))
print(value_to_bin(0.39, -0.4, 0.4, 10))

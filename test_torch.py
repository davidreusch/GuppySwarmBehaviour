import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyper_params import *
from guppy_model import *

torch.manual_seed(1)

torch.set_default_dtype(torch.float64)
path = "guppy_net.pth"
state_dict = torch.load(path)
for item in state_dict:
    print(item, state_dict[item].size())
model = LSTM_fixed()
#model.load_state_dict(torch.load(path))

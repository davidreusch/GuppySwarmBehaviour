import torch.nn as nn
import torch
from hyper_params import *
from auxiliary_funcs import *
from torch.distributions.categorical import Categorical

loss_function = nn.MSELoss()


# inspired by https://github.com/LeanManager/NLP-PyTorch/blob/master/Character-Level%20LSTM%20with%20PyTorch.ipynb
# TODO: Klassifizierung vs Regression
# TODO: Prediction, Training Error!
class LSTM_fixed(nn.Module):
    def __init__(self, input_size=input_dim, hidden_layer_size=hidden_layer_size):
        # output size has to be the number of bins for first loc vec component + for the second
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        # predict the two components
        self.linear = nn.Linear(hidden_layer_size, 2)
        self.hidden_state = None

    def forward(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        out = self.linear(x)
        return out, (h, c)

    def predict(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        out = self.linear(x)
        return out, (h, c)

    def init_hidden(self, batch_size, num_layers):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, self.hidden_layer_size).zero_(),
                weight.new(num_layers, batch_size, self.hidden_layer_size).zero_())

    def simulate(self, initial_pose, initial_loc_sensory, frames):
        pos = initial_pose[0], initial_pose[1]

        # for i in range(frames):


class LSTM_multi_modal(nn.Module):
    def __init__(self, input_size=input_dim, hidden_layer_size=hidden_layer_size):
        # output size has to be the number of bins for first loc vec component + for the second
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        self.linear1 = nn.Linear(hidden_layer_size, num_angle_bins)
        self.linear2 = nn.Linear(hidden_layer_size, num_speed_bins)

        # predict the two components
        self.hidden_state = self.init_hidden(batch_size, num_layers)

    def forward(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        angle_out = self.linear1(x)
        speed_out = self.linear2(x)
        return angle_out, speed_out, (h, c)


    def predict(self, x, hc):
        # not ready
        x, (h, c) = self.lstm(x, hc)
        angle_out = self.linear1(x)
        speed_out = self.linear2(x)
        angle_out = angle_out.view(num_angle_bins)
        speed_out = speed_out.view(num_speed_bins)
        m = nn.Softmax(0)
        angle_bin = Categorical(m(angle_out)).sample()
        speed_bin = Categorical(m(speed_out)).sample()
        angle_value = angle_bin_to_value(angle_bin, angle_min, angle_max, num_angle_bins, 0.01)
        speed_value = speed_bin_to_value(speed_bin, speed_min, speed_max, num_speed_bins, 0.01)
        return (angle_value, speed_value), (h, c)

    def init_hidden(self, batch_size, num_layers):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, self.hidden_layer_size).zero_(),
                weight.new(num_layers, batch_size, self.hidden_layer_size).zero_())

    def simulate(self, initial_pose, initial_loc_sensory, frames):
        pass

        # for i in range(frames):

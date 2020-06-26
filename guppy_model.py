import torch.nn as nn
from view_hdf import vec_to_angle

num_guppy_bins = 20
num_wall_rays = 20
input_dim = num_guppy_bins + num_wall_rays + 2
agent = 0
angle_bins = 10
speed_bins = 2  # take only 2 bins for the speed data which is constant in the simulated data
output_dim = angle_bins + speed_bins
num_layers = 1
hidden_layer_size = 100
batch_size = 4

loss_function = nn.MSELoss()


# inspired by https://github.com/LeanManager/NLP-PyTorch/blob/master/Character-Level%20LSTM%20with%20PyTorch.ipynb
# TODO: Klassifizierung vs Regression
# TODO: Prediction, Training Error!
class LSTM(nn.Module):
    def __init__(self, input_size=input_dim, hidden_layer_size=hidden_layer_size):
        # output size has to be the number of bins for first loc vec component + for the second
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # self.linear1 = nn.Linear(hidden_layer_size, angle_bins)
        # self.linear2 = nn.Linear(hidden_layer_size, speed_bins)

        # predict the two components
        self.linear = nn.Linear(hidden_layer_size, 2)
        self.hidden_state = self.init_hidden(batch_size, num_layers)

    def forward(self, x, hc):
        # print("Input seq: ", input_seq.view(1,1,len(input_seq)))
        # print("Hidden Cell: ", self.hidden_cell)

        x, (h, c) = self.lstm(x, hc)

        # angle_out = self.linear1(x)
        # speed_out = self.linear2(x)

        out = self.linear(x)

        self.hidden_state = h, c

        return out, (h, c)

    # return angle_out, speed_out, (h, c)

    def predict(self, test_ex, label):
        # not ready
        x, (h, c) = self.lstm(test_ex, self.hidden_state)
        out = self.linear(x)
        loss = loss_function(out, label)
        print(loss.item())

    def init_hidden(self, batch_size, num_layers):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, self.hidden_layer_size).zero_(),
                weight.new(num_layers, batch_size, self.hidden_layer_size).zero_())

    def simulate(self, initial_pose, initial_loc_sensory, frames):
        pos = initial_pose[0], initial_pose[1]
        ori = vec_to_angle(initial_pose[2], initial_pose[3])

        #for i in range(frames):







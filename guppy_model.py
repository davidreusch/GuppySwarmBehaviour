import torch.nn as nn
import torch
from hyper_params import *
from auxiliary_funcs import *
from torch.distributions.categorical import Categorical


# inspired by https://github.com/LeanManager/NLP-PyTorch/blob/master/Character-Level%20LSTM%20with%20PyTorch.ipynb
# TODO: Eyolfsdottir korrigieren
class LSTM_fixed(nn.Module):
    def __init__(self, optimizer=None, input_dim=input_dim, hidden_layer_size=hidden_layer_size, arch=arch ):
        super().__init__()


        # # predict the two components
        self.lossfn = nn.MSELoss()
        self.linear = nn.Linear(hidden_layer_size, 2)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm([hidden_layer_size])
        #
        if arch != "ey":
            self.lstm = nn.LSTM(input_dim, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.dis_layers = nn.ModuleList([nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True )
                                              for _ in range(num_layers - 1)])
            self.dis_layers.insert(0, nn.LSTM(input_dim, hidden_layer_size, 1, batch_first=True))

            self.gen_layers = nn.ModuleList([nn.LSTM(hidden_layer_size * 2, hidden_layer_size, 1, batch_first=True)
                                                for _ in range(num_layers - 1)])
            self.gen_layers.append(nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True))

        #self.layernorm_dis = nn.LayerNorm(hidden_layer_size)
        #self.layernorm_gen = nn.LayerNorm(hidden_layer_size * 2)

    def forward(self, x, hidden):
        if arch == "ey":
            return self.forwardey(x, hidden)
        else:
            return self.forwardold(x, hidden)

    def forwardold(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        x = self.layernorm(x)
        x = self.dropout(x)
        out = self.linear(x)
        return out, (h, c)

    def forwardey(self, x, states):
        dis_states, gen_states = states[: num_layers], states[num_layers:][::-1]
        seq_len = x.size()[1]

        # discriminative network applies the lstm layers and saves the results
        # and hidden state
        layer_results = [self.dis_layers[0](x, dis_states[0])]
        for l in range(1, num_layers):
            layer_results.append(
                self.dis_layers[l](
                    ((layer_results[-1][0])),
                    dis_states[l]
                )
            )
        # don't need this here but maybe later
        # dis_out = (layer_results[-1][0] + 1.0) / 2.0

        # generative network gets as input the previous input plus an input from the discriminative layer
        top_layer = num_layers - 1
        layer_results.append(self.gen_layers[top_layer]((layer_results[-1][0]), gen_states[top_layer]))
        for l in range(top_layer - 1, -1, -1):
            # next = torch.cat((layer_results[-1][0], layer_results[l][0]), 2)
            # print(next.size())
            layer_results.append(
                self.gen_layers[l](
                    ((torch.cat((layer_results[-1][0], layer_results[l][0]), 2))),
                    gen_states[l]
                )
            )

        # transform the output into bins
        gen_out = layer_results[-1][0]
        out = self.linear(gen_out)

        # get the whole hidden state history
        next_states = [results[1] for results in layer_results]
        return out, next_states

    def predict(self, x, hc):
        return self.forward(x, hc)


    def do_batch(self, inputs, targets, batch_size=batch_size):
        states = self.init_hidden(batch_size, num_layers, hidden_layer_size)
        prediction, states = self.forward(inputs, states)
        loss = self.lossfn(prediction, targets)
        return prediction, states, loss

    def init_hidden(self, batch_size, num_layers, hidden_layer_size):
        return [self.init_hiddenh(batch_size, 1, hidden_layer_size) for _ in
                range(num_layers * 2)] if arch == "ey" \
            else self.init_hiddenh(batch_size, num_layers, hidden_layer_size)

    def init_hiddenh(self, batch_size, num_layers, hidden_layer_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, hidden_layer_size).zero_(),
                weight.new(num_layers, batch_size, hidden_layer_size).zero_())

class LSTM_multi_modal(nn.Module):
    def __init__(self, input_size=input_dim, hidden_layer_size=hidden_layer_size, arch=arch):
        # output size has to be the number of bins for first loc vec component + for the second
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        #self.linear1 = nn.Linear(hidden_layer_size, num_angle_bins)
        #self.linear2 = nn.Linear(hidden_layer_size, num_speed_bins)

        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm([hidden_layer_size])

        if arch != "ey":
            self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)
            self.linear_inter = nn.Linear(hidden_layer_size, hidden_layer_size)
            self.activation = nn.ReLU()
            self.linear_out = nn.Linear(hidden_layer_size, num_angle_bins + num_speed_bins)
            self.lossfn = nn.CrossEntropyLoss()
        else:
            self.dis_layers = nn.ModuleList([nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True )
                                for _ in range(num_layers - 1)])
            self.dis_layers.insert(0, nn.LSTM(input_size, hidden_layer_size, 1, batch_first=True))
            #
            self.gen_layers = nn.ModuleList([nn.LSTM(hidden_layer_size * 2, hidden_layer_size, 1, batch_first=True)
                                 for _ in range(num_layers - 1)])
            self.gen_layers.append(nn.LSTM(hidden_layer_size, hidden_layer_size, 1, batch_first=True))
            #
            self.dropout = nn.Dropout(dropout)
            #self.layernorm_dis = nn.LayerNorm(hidden_layer_size)
            #self.layernorm_gen = nn.LayerNorm(hidden_layer_size * 2)

    def forward(self, x, hidden):
        if arch == "ey":
            return self.forwardey(x, hidden)
        else:
            return self.forwardff(x, hidden)

    def forwardff(self, x, hc):
        x, (h, c) = self.lstm(x, hc)
        x = self.layernorm1(x)
        x = self.dropout(x)
        x = self.linear_inter(x)
        x = self.activation(x)
        out = self.linear_out(x)


        #x = m(x)
        #angle_out = self.linear1(x)
        #speed_out = self.linear2(x)

        return out, (h, c)

    # from Moritz Maxeiner, eyolfsdottirs method
    def forwardey(self, x, states):
        dis_states, gen_states = states[: num_layers], states[num_layers:][::-1]
        seq_len = x.size()[1]

        # discriminative network applies the lstm layers and saves the results
        # and hidden state
        layer_results = [(self.dis_layers[0](x, dis_states[0]))]
        for l in range(1, num_layers):
            layer_results.append(
                self.dis_layers[l](
                    self.dropout((layer_results[-1][0])),
                    dis_states[l]
                )
            )
        # don't need this here but maybe later
        # dis_out = (layer_results[-1][0] + 1.0) / 2.0

        # generative network gets as input its the previous input plus an input from the discriminative layer
        top_layer = num_layers - 1
        layer_results.append(self.gen_layers[top_layer]((layer_results[-1][0]), gen_states[top_layer]))
        for l in range(top_layer - 1, -1, -1):
            #next = torch.cat((layer_results[-1][0], layer_results[l][0]), 2)
            #print(next.size())
            layer_results.append(
                self.gen_layers[l](
                    self.dropout((torch.cat((layer_results[-1][0], layer_results[l][0]), 2))),
                    gen_states[l]
                )
            )

        # transform the output into bins
        gen_out = layer_results[-1][0]
        #out = self.linear(gen_out)
        angle_out = self.linear1(gen_out)
        speed_out = self.linear2(gen_out)

        # get the whole hidden state history
        next_states = [results[1] for results in layer_results]
        return angle_out, speed_out, next_states
        #return out, next_states


    def predict(self, x, states):
        # works only with batchsize = sequencesize = 1
        #angle_out, speed_out, states = self.forward(x, states)
        out, states = self.forward(x, states)
        out = out.view(-1)
        angle_out = out[:num_angle_bins]
        speed_out = out[num_angle_bins:]
        # print("-----------RAW SCORES------------")
        # print(angle_out)
        # print(speed_out)
        m = nn.Softmax(0)
        angle_prob = m(angle_out)
        speed_prob = m(speed_out)
        # print("---------------SOFTMAX PROBABILITIES ---------------")
        # print(angle_prob)
        # print(speed_prob)
        angle_bin = Categorical(angle_prob).sample()
        speed_bin = Categorical(speed_prob).sample()
        # print("angle bin predicted: ", angle_bin)
        # print("speed bin predicted: ", speed_bin)
        angle_value = angle_bin_to_value(angle_bin, angle_min, angle_max, num_angle_bins, 0.001)
        speed_value = speed_bin_to_value(speed_bin, speed_min, speed_max, num_speed_bins, 0.001)
        # print("angle value:", angle_value)
        # print("speed value:", speed_value)
        return (angle_value, speed_value), states

    def do_batch(self, inputs, targets, batch_size=batch_size, verbose=False):
        targets = targets.type(torch.LongTensor)
        states = self.init_hidden(batch_size, num_layers, hidden_layer_size)

        # forward
        pred, states = self.forward(inputs, states)

        # reshape
        pred = pred.view(pred.shape[0] * pred.shape[1], -1)
        angle_pred = pred[:, :num_angle_bins]
        speed_pred = pred[:, num_angle_bins:]
        targets = targets.view(targets.shape[0] * targets.shape[1], -1)
        angle_targets = targets[:, 0]
        speed_targets = targets[:, 1]

        # calculate loss
        loss1 = self.lossfn(angle_pred, angle_targets)
        loss2 = self.lossfn(speed_pred, speed_targets)
        loss = (loss1 + loss2) / 2

        if verbose:
            torch.set_printoptions(threshold=10000)
            with torch.no_grad():
                for j in range(1):
                    angle_probs = nn.Softmax(0)(angle_pred[j])
                    speed_probs = nn.Softmax(0)(speed_pred[j])
                    print("angle prob:\n", angle_probs[angle_targets[j].data])
                    print(angle_targets[j])

        return pred, states, loss

    def init_hidden(self, batch_size, num_layers, hidden_layer_size):
        return [self.init_hiddenh(batch_size, 1, hidden_layer_size) for _ in
                  range(num_layers * 2)] if arch == "ey" \
            else self.init_hiddenh(batch_size, num_layers, hidden_layer_size)

    def init_hiddenh(self, batch_size, num_layers, hidden_layer_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, hidden_layer_size).zero_(),
            weight.new(num_layers, batch_size, hidden_layer_size).zero_())



























    def predict_old(self, x, hc):
        # works only with batchsize = sequencesize = 1
        x, (h, c) = self.lstm(x, hc)
        angle_out = self.linear1(x)
        speed_out = self.linear2(x)
        angle_out = angle_out.view(num_angle_bins)
        speed_out = speed_out.view(num_speed_bins)
        m = nn.Softmax(0)
        angle_prob = m(angle_out)
        speed_prob = m(speed_out)
        print(angle_prob)
        print(speed_prob)
        angle_bin = Categorical(angle_prob).sample()
        speed_bin = Categorical(speed_prob).sample()
        angle_value = angle_bin_to_value(angle_bin, angle_min, angle_max, num_angle_bins, 0.001)
        speed_value = speed_bin_to_value(speed_bin, speed_min, speed_max, num_speed_bins, 0.001)
        return (angle_value, speed_value), (h, c)

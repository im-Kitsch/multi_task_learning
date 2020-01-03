import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from torch.utils.tensorboard import SummaryWriter


class RnnNet(nn.Module):
    def __init__(self, input_size, hid_size, shared_layers):
        super(RnnNet, self).__init__()
        self.gru_1 = nn.GRU(
            input_size=input_size,
            hidden_size=hid_size,
            num_layers=1,
            bias=True
        )

        self.gru_mid1, self.gru_mid2 = shared_layers

        self.gru_out = nn.GRU(
            input_size=hid_size,
            hidden_size=hid_size,
            num_layers=1,
            bias=True
        )
        self.dense_out = nn.Linear(hid_size, input_size)

        return

    def forward(self, x, y):
        output1, hid1 = self.gru_1(x)

        _, hid2 = self.gru_mid1(output1)

        output3, hid3 = self.gru_mid2(y, hid1)

        output4, hid4 = self.gru_out(output3, hid2)

        predict = self.dense_out(output4)

        return predict


class DataSampler:
    def __init__(self, original_data, discrete_num_axis):
        self.original_data = original_data

        self.code_dim = discrete_num_axis * discrete_num_axis

        interval = np.linspace(0, 1, discrete_num_axis).reshape(1, -1)

        x, y = original_data[:, 0], original_data[:, 1]
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)

        index_x = (x >= interval[0, :discrete_num_axis-1]) * (x <= interval[0, 1:])
        index_x = np.argmax(index_x, axis=1)

        index_y = (y >= interval[0, :discrete_num_axis - 1]) * (y <= interval[0, 1:])
        index_y = np.argmax(index_y, axis=1)

        index = index_y * discrete_num_axis + index_x

        self.coded_data = index.reshape(-1, 1)
        self.interval = interval
        self.code_dict = (interval[1:] + interval[:discrete_num_axis-1]) / 2

        ohe = OneHotEncoder(categories='auto')
        ohe.fit(self.coded_data)
        data_one_hot = ohe.transform(self.coded_data).toarray()

        self.one_hot_data = data_one_hot
        self.ohe = ohe
        self.one_hot_dim = self.one_hot_data.shape[1]
        return

    def sample(self, num, batch_len):
        start_ind = np.random.randint(0, self.original_data.shape[0]-batch_len, num)

        batch_data = np.zeros((batch_len, num, self.one_hot_dim))
        for i in range(num):
            batch_data[:, i, :] = self.one_hot_data[start_ind[i]:start_ind[i]+batch_len]

        # batch_data_code = self.ohe.inverse_transform(batch_data.reshape(-1, self.one_hot_dim))
        return batch_data

    def decode(self, data):

        return


def train_step(rnn_net, batch_data, optimizer):
    batch_input = batch_data[:30, :, :]
    batch_label = batch_data[30:, :, :]
    optimizer.zero_grad()
    batch_predict = rnn_net(batch_input, batch_label)
    batch_predict = batch_predict.view(-1, one_hot_dim)

    batch_label = batch_label.view(-1, one_hot_dim)
    _, predict_code = torch.max(batch_label, dim=1)
    loss = F.cross_entropy(batch_predict.view(-1, one_hot_dim),
                           predict_code.view(-1))
    loss.backward()
    optimizer.step()
    return loss

def fig2rgb_array(fig, expand=False):
  fig.canvas.draw()
  buf = fig.canvas.tostring_rgb()
  ncols, nrows = fig.canvas.get_width_height()
  shape = (3, nrows, ncols) if not expand else (1, 3, nrows, ncols)
  return np.frombuffer(buf, dtype=np.uint8).reshape(shape)

data_num = 3000
HID_SIZE = 64

theta = np.linspace(0, 2.1 * np.pi, data_num)

original_data = np.zeros((data_num, 2))
original_data[:, 0] = np.sin(theta)
original_data[:, 1] = np.cos(theta)

data_sampler = DataSampler(original_data, 25)
one_hot_dim = data_sampler.one_hot_dim


share_gru1 = nn.GRU(input_size=HID_SIZE,
                    hidden_size=HID_SIZE,
                    num_layers=1,
                    bias=True)

share_gru2 = nn.GRU(input_size=one_hot_dim,
                    hidden_size=HID_SIZE,
                    num_layers=1,
                    bias=True)

rnn_net_1 = RnnNet(one_hot_dim, HID_SIZE, [share_gru1, share_gru2])
rnn_net_2 = RnnNet(one_hot_dim, HID_SIZE, [share_gru1, share_gru2])

optimizer_1 = torch.optim.Adam(rnn_net_1.parameters(), lr=1e-3)
optimizer_2 = torch.optim.Adam(rnn_net_2.parameters(), lr=1e-3)

writer = SummaryWriter(comment="RNN")

for _i in range(1000):
    batch_data = data_sampler.sample(num=15, batch_len=50)
    batch_data = torch.FloatTensor(batch_data)
    loss1 = train_step(rnn_net_1, batch_data, optimizer_1)

    batch_data = data_sampler.sample(num=15, batch_len=50)
    batch_data = torch.FloatTensor(batch_data)
    batch_data = batch_data.flip(0)
    loss2 = train_step(rnn_net_2, batch_data, optimizer_2)

    writer.add_scalar("loss/net1", loss1, _i)
    writer.add_scalar("loss/net2", loss2, _i)
    print(f"step {_i}, loss {loss1} {loss2}")

    if _i%100 == 0:
        pass # to visualize the trajectory




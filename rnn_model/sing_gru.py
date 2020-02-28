#%%
import numpy as np

import torch
import torchsummaryX
from torch.utils.tensorboard import SummaryWriter

import argparse

from util.traj_dataloader import TrajDataset
from util.coordi_map import parse_MeshCode
# from plot_util import plot_with_arrow
from util.plot_util import visualization_trajectory

from util.nlp_util import random_choose_topk


class GruNN(torch.nn.Module):
    def __init__(self, num_class, hid_size, hid_layers,
                 drop_out_rate, embeding_size=None):

        super(GruNN, self).__init__()

        self.embedding = torch.nn.Embedding(num_class, embeding_size)
        self.gru = torch.nn.GRU(
            input_size=embeding_size,
            hidden_size=hid_size,
            num_layers=hid_layers,
            bias=True,
            dropout=drop_out_rate)

        self.dense = torch.nn.Linear(hid_size, num_class)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)

        self.num_class = num_class
        return

    def forward(self, input_data, h_0=None):

        assert input_data.ndim == 2

        input_data = self.embedding(input_data)

        if h_0 is None:
            output, h_n = self.gru(input_data)
        else:
            output, h_n = self.gru(input_data, h_0)

        output = self.dense(output)
        output = self.log_softmax(output)
        return output, h_n

    def predict_next(self, input_data, h_n=None, top_k=5, look_back_n=1, sample_policy=None):

        pred_logits, hid_st = self.forward(input_data, h_n)

        pred_logits = pred_logits[-look_back_n:]

        choice_val, choice_ix = random_choose_topk(pred_logits, top_k)
        return choice_val, choice_ix, hid_st

    def generate_traj(self, begin_data, traj_len, batch_first, h_n=None):
        assert begin_data.ndim == 2
        assert batch_first is False

        _, batch_size = begin_data.shape
        gen_traj = torch.zeros(traj_len, batch_size, dtype=int)

        traj_ix = begin_data
        for _i in range(traj_len):
            _, traj_ix, h_n = self.predict_next(traj_ix, h_n, top_k=3, look_back_n=1)
            gen_traj[_i, :] = traj_ix.flatten()
        return gen_traj

    def fit(self, training_data, training_label,
              optimizer, writer, batch_size=64, n_epochs=10,
              device=torch.device("cuda:0")):
        training_dataset = TrajDataset(training_data, training_label)

        training_dataloader = torch.utils.data.DataLoader(
            training_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4,
            drop_last=False)

        predict_dim = training_label.shape[1]
        log_dir = writer.log_dir
        total_samples = training_dataloader.dataset.__len__() #TODO need to be len()
        criterion = torch.nn.NLLLoss(reduction="mean")

        total_step = 0
        for _epoc in range(n_epochs):
            for i_batch, batch_data in enumerate(training_dataloader):
                data_x, data_y = batch_data
                data_x, data_y = data_x.T, data_y.T

                data_x, data_y = data_x.to(device), data_y.to(device)

                data_predict, _ = self.forward(data_x)
                data_predict = data_predict[-predict_dim:, :]
                data_predict = data_predict.reshape(-1, self.num_class)
                data_y = data_y.reshape(-1)

                optimizer.zero_grad()
                loss = criterion(data_predict, data_y)
                loss.backward()
                optimizer.step()

                if i_batch % 5 == 0:
                    print(f"epoch {_epoc} ",
                          f"{i_batch * training_dataloader.batch_size / total_samples * 100:.2f}% ",
                          f"{loss}")

                writer.add_scalar("loss/training_loss", loss, total_step)
                total_step += 1

            torch.save(self.state_dict(), f"{log_dir}/chech_point_{_epoc}.pth")
        return

    # need to be set as eval
    def eval_sample_traj_visualize(self, traj_length, begin_code, n_r, n_c, trans_ix2coordi, batch_first):
        assert batch_first is False

        gen_traj = self.generate_traj(begin_code, traj_length, batch_first=False)
        gen_traj = torch.cat([begin_code, gen_traj], dim=0)
        gen_traj = gen_traj.numpy()
        gen_traj = trans_ix2coordi(gen_traj)

        gen_traj = np.swapaxes(gen_traj, 0, 1)
        fig1, fig2 = visualization_trajectory(gen_traj, n_r, n_c)
        return fig1, fig2

    def eval_perplexity(self, data, batch_first):
        assert batch_first is False

        input_data = data[:-1]
        label = data[1:]

        val_loss = 0.
        counter = 0
        n_sample = data.shape[1]

        while counter < n_sample:
            _dt, _lbl = input_data[:, counter:counter+500], label[:, counter:counter+500]
            _dt, _lbl = torch.tensor(_dt), torch.tensor(_lbl)
            logits, _ = self.forward(_dt)

            _lbl = _lbl .flatten()
            _len_lbl = len(_lbl)
            loss = torch.nn.functional.nll_loss(logits.reshape(_len_lbl, -1), _lbl)
            val_loss += loss.item()
        val_loss = torch.exp(val_loss)
        return val_loss






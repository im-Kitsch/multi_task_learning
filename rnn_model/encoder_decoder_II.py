# %%
import numpy as np
import math

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

from util.traj_dataloader import TrajDataset
from util.coordi_map import parse_MeshCode
# from plot_util import plot_with_arrow
from util.plot_util import visualization_trajectory

from util.nlp_util import random_choose_topk, add_start_num_np


class EncoderDecoderII(torch.nn.Module):
    def __init__(self, num_class, hid_size, hid_layers,
                 drop_out_rate, embeding_size):
        super(EncoderDecoderII, self).__init__()
        self.num_class = num_class

        self.encoder_embedding = torch.nn.Embedding(num_class, embeding_size)
        self.encoder_gru = torch.nn.GRU(embeding_size, hid_size, hid_layers,
                                        bias=True, dropout=drop_out_rate)

        self.decoder_gru = torch.nn.GRU(embeding_size, hid_size, hid_layers,
                                        bias=True, dropout=drop_out_rate)
        self.decoder_dense = torch.nn.Linear(hid_size, num_class)
        self.decoder_logsoftmax = torch.nn.LogSoftmax(dim=2)
        return

    def forward(self, input_data, refer_label, h_0=None, batch_first=False):
        if batch_first is True:
            input_data = torch.transpose(input_data, 0, 1)
            refer_label = torch.transpose(refer_label, 0, 1)

        _, h_en = self.__encoder_forward(input_data, h_0)
        out, h_de = self.__decoder_forward(refer_label, h_en)
        return out, h_de

    def __encoder_forward(self, input_data, h_en0):
        out = self.encoder_embedding(input_data)
        out, h_en = self.encoder_gru(out, h_en0)
        return out, h_en

    def __decoder_forward(self, refer_label, h_de_0):
        refer_label_embeded = self.encoder_embedding(refer_label) #TODO share the same embedding?
        out, h_de = self.decoder_gru(refer_label_embeded, h_de_0)
        out = self.decoder_dense(out)
        out = self.decoder_logsoftmax(out)
        return out, h_de

    def fit(self, training_data, training_label, validation_data, validation_label,
            optimizer, writer, start_num, batch_size=64, n_epochs=10,
            device=torch.device("cuda:0")):

        # Attention, assuming batch first for data

        num_data, seq_len_label = training_label.shape[0], training_label.shape[1]

        training_dataloader = torch.utils.data.DataLoader(
                                TrajDataset(training_data,
                                            add_start_num_np(training_label,
                                                             start_num=start_num,
                                                             repeat_time=1, batch_first=True)),
                                batch_size=batch_size, shuffle=True,
                                num_workers=4, drop_last=False) #TODO specify batchfirst

        log_dir = writer.log_dir
        total_samples = len(training_dataloader.dataset)

        criterion = torch.nn.NLLLoss(reduction="mean")

        total_step = 0
        for _epoc in range(n_epochs):
            for i_batch, (data_x, data_y) in enumerate(training_dataloader):
                data_x, data_y = data_x.T, data_y.T
                data_x, data_y = data_x.to(device), data_y.to(device)

                pred_logits, _ = self.forward(data_x, data_y)
                pred_logits = pred_logits.reshape(-1, self.num_class)
                data_y = data_y.flatten()

                optimizer.zero_grad()
                loss = criterion(pred_logits, data_y)
                loss.backward()
                optimizer.step()

                writer.add_scalar("loss/training_loss", loss, total_step)
                total_step += 1

                if i_batch % 5 == 0:
                    print(f"epoch {_epoc} ",
                          f"{i_batch * training_dataloader.batch_size / total_samples * 100:.2f}% ",
                          f"{loss}")

            with torch.no_grad():
                self.eval()
                self.to(torch.device("cpu"))
                train_loss = self.eval_perplexity(validation_data, validation_label, device=torch.device("cpu"),
                                                  start_num=start_num, batch_first=True)
                val_loss = self.eval_perplexity(training_data, training_label, device=torch.device("cpu"),
                                                start_num=start_num, batch_first=True)
                fig = self.eval_relativity(validation_data, validation_label, start_num,
                                           batch_first=True, device=torch.device("cpu"),
                                           count_interval=4)
                writer.add_figure("relativity", fig)
            print(f"------------"
                  f"{_epoc}, train_loss {train_loss} validation_loss {val_loss}"
                  f"------------")
            torch.save(self.state_dict(), f"{log_dir}/chech_point_{_epoc}.pth")

            self.train()
            self.to(device)
        return

    def predict_next(self, ref_label, h_de, top_k=5,
                     look_back_n=1, sample_policy=None, batch_first=False):
        assert batch_first is False

        pred_logits, hid_de = self.__decoder_forward(ref_label, h_de)
        pred_logits = pred_logits[-look_back_n:]

        choice_val, choice_ix = random_choose_topk(pred_logits, top_k)

        return choice_val, choice_ix, hid_de

    def generate_traj(self, begin_data, traj_len, batch_first,
                      begin_num, topk=3, h_en=None):
        assert begin_data.ndim == 2
        if batch_first is True:
            begin_data = begin_data.T

        _, batch_size = begin_data.shape
        gen_traj = torch.zeros(traj_len+1, batch_size, dtype=int)
        gen_traj[0, :] += begin_num

        _, h_de = self.__encoder_forward(begin_data, h_en0=h_en)
        traj_i = gen_traj[[0], :]
        for i in range(traj_len):
            traj_i = traj_i.reshape(1, -1)
            _, traj_i, h_de = self.predict_next(ref_label=traj_i, h_de=h_de,
                                                top_k=topk, look_back_n=1, batch_first=False)
            gen_traj[i+1, :] = traj_i.flatten()

        return gen_traj[1:]

    # need to be set as eval
    def eval_sample_traj_visualize(self, traj_length, begin_code,
                                   n_r, n_c, trans_ix2coordi, batch_first):
        return fig1, fig2

    def eval_perplexity(self, data, refer_label, start_num, batch_first, device):
        if batch_first is True:
            data, refer_label = data.T, refer_label.T
        refer_label = add_start_num_np(input_dt=refer_label, start_num=start_num,
                                       repeat_time=1, batch_first=False)
        data, refer_label = torch.tensor(data), torch.tensor(refer_label)
        data, refer_label = data.to(device), refer_label.to(device)
        pred_logits, _ = self.forward(data, refer_label)
        pred_logits = pred_logits.reshape(-1, self.num_class)
        refer_label = refer_label.flatten()
        val_loss = torch.nn.functional.nll_loss(pred_logits, refer_label,
                                            reduction="mean")
        return torch.exp(val_loss)

    # TODO, relativity coefficient
    def eval_relativity(self, data, refer_label, start_num, batch_first,
                        device, count_interval):
        if batch_first is True:
            data, refer_label = data.T, refer_label.T
        # refer_label = add_start_num_np(input_dt=refer_label, start_num=start_num,
        #                                repeat_time=1, batch_first=False)
        data, refer_label = torch.tensor(data), torch.tensor(refer_label)
        data, refer_label = data.to(device), refer_label.to(device)

        gen_traj = self.generate_traj(data, refer_label.shape[0], batch_first=False,
                                 begin_num=start_num, topk=3)

        estimated = torch.cat([data, gen_traj], dim=0)
        ground_truth = torch.cat([data, refer_label], dim=0)

        batch_size = data.shape[1]

        reference_ix = np.arange(self.num_class).reshape(1, 1, 1, -1)
        estimated = estimated.numpy()
        ground_truth = ground_truth.numpy()
        estimated = estimated.reshape(-1, count_interval, batch_size, 1)
        ground_truth = ground_truth.reshape(-1, count_interval, batch_size, 1)

        gen_traj_stats = np.sum(estimated == reference_ix, axis=(1, 2))
        ground_truth_stats = np.sum(ground_truth == reference_ix, axis=(1, 2))

        num_plots = gen_traj_stats.shape[0]
        n_c = 3
        n_r = math.ceil(num_plots / n_c)
        fig, axes = plt.subplots(nrows=n_r, ncols=n_c)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            sns.regplot(x=gen_traj_stats[i], y=ground_truth_stats[i], ax=ax)
        return fig

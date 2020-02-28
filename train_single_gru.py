import numpy as np

import torch
import torchsummaryX
from torch.utils.tensorboard import SummaryWriter
from rnn_model.sing_gru import GruNN

import argparse

from util.traj_dataloader import TrajDataset, PflowLoader
from util.nlp_util import eval_relativity

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--embed-size', type=int, default=40, metavar='N',
                    help='hidden layer')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--hid-size', type=int, default=80, metavar='N',
                    help='hidden size')
parser.add_argument('--hid-layer', type=int, default=2, metavar='N',
                    help='hidden layer')
parser.add_argument('--drop-out', type=float, default=0.2, metavar='DR_R',
                    help='drop out rate')
parser.add_argument('--random-seed', type=float, default=0, metavar='RAND_SEED',
                    help='drop out rate')
args, unknown = parser.parse_known_args()
N_EPOCH = args.epochs
BATCH_SIZE = args.batch_size
HID_SIZE = args.hid_size
HID_LAYER = args.hid_layer
LR = args.lr
DROP_OUT_RATE = args.drop_out
EMBEDDING_DIM = args.embed_size
MANUAL_SEED = args.random_seed

np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

PFLOW_DATA_PATH = "~/data/pflow_mini_preprocessed.csv"

writer = SummaryWriter(comment="single_gru")

writer.add_hparams({"epoch":N_EPOCH, "batch_size":BATCH_SIZE,
                    "hid_size":HID_SIZE, "hid_layer":HID_LAYER,
                    "learning_rate":LR, "drop_out":DROP_OUT_RATE,
                    "embedding_dim":EMBEDDING_DIM, "random seed": MANUAL_SEED}, {})

device = torch.device("cuda:0")
#%%
print("-----------loading data---------")
pflow_loader = PflowLoader(PFLOW_DATA_PATH, "./dict_file", begin_ix=480, time_interval=30, num_per_day=24,
                 mode="normal", require="traj_code", begin_code=None)

dataset, num_class = pflow_loader.get_data()
print("---------loading data finished---------")
num_traning_sample = int(dataset.shape[0] * 0.8)
training_data, training_label = dataset[:num_traning_sample, :-1], dataset[:num_traning_sample, 1:]
valiation_data, valiation_label = dataset[num_traning_sample:, :-1], dataset[num_traning_sample:, 1:]
traj_len = dataset.shape[1]
print("---------prepare data finished----------")
print(f"training data {num_traning_sample} samples ")
print(f"validation data {valiation_data.shape[0]} sample")
print(f"trainin data length {training_data.shape[1]}, label length {training_label.shape[1]} data length {traj_len}")
# TODO del dataset

gru_nn = GruNN(num_class=num_class, hid_size=HID_SIZE, hid_layers=HID_LAYER,
                 drop_out_rate=DROP_OUT_RATE, embeding_size=EMBEDDING_DIM)
optimizer = torch.optim.Adam(gru_nn.parameters(), lr=LR)

gru_nn.to(device)

gru_nn.fit(training_data, training_label, optimizer=optimizer,
           writer=writer, batch_size=BATCH_SIZE, n_epochs=N_EPOCH,
           device=device)

#%%
# evaluation
gru_nn.to("cpu")

begin_code = dataset[:9, :4]
begin_code = begin_code.T
begin_code = torch.tensor(begin_code)
fig1, fig2 = gru_nn.eval_sample_traj_visualize(traj_length=traj_len-4, begin_code=begin_code, n_r=3, n_c=3,
                                               trans_ix2coordi=pflow_loader.trans_ix2coordi, batch_first=False)
writer.add_figure("generate_fig_2d", fig1)
writer.add_figure("generate_fig_3d", fig2)

val_dataset = dataset[num_traning_sample:]
val_dataset = val_dataset.T
fig = eval_relativity(gru_nn, val_dataset, use_len=4, count_interval=4, num_class=num_class, batch_first=False)
writer.add_figure("relativity", fig)




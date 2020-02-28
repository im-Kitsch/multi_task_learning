import numpy as np
import torch
import torchsummaryX
from torch.utils.tensorboard import SummaryWriter
from rnn_model.sing_gru import GruNN

import argparse

from util.traj_dataloader import TrajDataset, PflowLoader
from util.nlp_util import eval_relativity

from rnn_model.encoder_decoder_II import EncoderDecoderII

parser = argparse.ArgumentParser(description='PyTorch  Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
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

START_STR = "00000000"

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
pflow_loader = PflowLoader(PFLOW_DATA_PATH, "./dict_file",
                           begin_ix=480, time_interval=30, num_per_day=24,
                            mode="normal", require="traj_code", begin_code="00000000")
START_NUM = pflow_loader.trans_code2ix(np.array(START_STR, dtype="U8")).item()

dataset, num_class = pflow_loader.get_data()
print("---------loading data finished---------")
num_traning_sample = int(dataset.shape[0] * 0.8)
training_data, training_label = dataset[:num_traning_sample, :10], dataset[:num_traning_sample, 10:]
validation_data, validation_label = dataset[num_traning_sample:, :10], dataset[num_traning_sample:, 10:]
sample_len = dataset.shape[1]
print("---------prepare data finished----------")
print(f"training data {num_traning_sample} samples ")
print(f"validation data {validation_data.shape[0]} sample")
print(f"trainin data length {training_data.shape[1]}, label length {training_label.shape[1]} data length {sample_len}")
# TODO del dataset

#%%
rnn_en_de = EncoderDecoderII(num_class=num_class, hid_size=HID_SIZE, hid_layers=HID_LAYER,
                            drop_out_rate=DROP_OUT_RATE, embeding_size=EMBEDDING_DIM)
optimizer = torch.optim.Adam(rnn_en_de.parameters(), lr=LR)

rnn_en_de.to(device)

rnn_en_de.fit(training_data, training_label, validation_data, validation_label,
              optimizer=optimizer,
              writer=writer, start_num=START_NUM, batch_size=BATCH_SIZE,
              n_epochs=N_EPOCH, device=device)

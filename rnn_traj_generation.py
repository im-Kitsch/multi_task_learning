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


class GRU_NN(torch.nn.Module):
    def __init__(self, input_size, hid_size, hid_layers, output_size,
                 drop_out_rate, num_class, embeded_size=None):

        super(GRU_NN, self).__init__()

        if embeded_size is None:
            self.do_embedding = False
            self.gru = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=hid_layers,
                bias=True,
                dropout=drop_out_rate
            )
        else:
            self.do_embedding = True
            self.embed_lay = torch.nn.Embedding(input_size, embeded_size)
            self.gru = torch.nn.GRU(
                input_size=embeded_size,
                hidden_size=hid_size,
                num_layers=hid_layers,
                bias=True,
                dropout=drop_out_rate
            )

        self.num_class = num_class

        self.nn = torch.nn.Linear(hid_size, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
        return

    def forward(self, input_data, h_n=None):
        if self.do_embedding is False:
            _n_row, _n_col = input_data.shape
            input_data = input_data.reshape(-1)
            input_data = one_hot_transfer(input_data, self.num_class)
            input_data = input_data.reshape(_n_row, _n_col, self.num_class)
        else:
            input_data = self.embed_lay(input_data)

        if h_n is None:
            output, h_n = self.gru(input_data)
        else:
            output, h_n = self.gru(input_data, h_n)

        output = self.nn(output)
        output = self.log_softmax(output)
        return output, h_n

    def predict_next(self, input_data, h_n=None, top_k=5, sample_policy=None):

        pred_logits, hid_st = self.forward(input_data, h_n)
        pred_logits = pred_logits[-1]

        _, pred_topk = torch.topk(pred_logits, top_k)
        choice_inx = torch.randint(0, top_k, (pred_topk.shape[0],  1))
        choice_inx = pred_topk.gather(1, choice_inx)
        choice_inx = choice_inx.squeeze(1)
        return choice_inx, hid_st

    def generate_traj(self, begin_signal, traj_len, h_n=None):
        _, batch_size = begin_signal.shape
        gen_traj = torch.zeros(traj_len, batch_size)

        for _i in range(traj_len):
            gen_traj[_i, :], h_n = self.predict_next(begin_signal, h_n)
        return gen_traj


def evaluation(model, traj_length, begin_code, n_r, n_c, drop_num, dict_ix2words):
    model.eval()
    gen_traj = model.generate_traj(begin_code, traj_length)
    gen_traj = gen_traj[drop_num:]

    gen_traj = gen_traj.T
    gen_traj = gen_traj.numpy().astype(int)

    traj_gen_coordi = np.zeros((gen_traj.shape[0], gen_traj.shape[1], 2))
    for i in range(gen_traj.shape[0]):
        for j in range(gen_traj.shape[1]):
            code = gen_traj[i, j]
            traj_gen_coordi[i, j, 0], traj_gen_coordi[i, j, 1] = parse_MeshCode(dict_ix2words[code])
    fig1, fig2 = visualization_trajectory(traj_gen_coordi, n_r, n_c)
    return fig1, fig2


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=4, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--hid-size', type=int, default=80, metavar='N',
                    help='hidden size')
parser.add_argument('--hid-layer', type=int, default=2, metavar='N',
                    help='hidden layer')
parser.add_argument('--embed-size', type=int, default=40, metavar='N',
                    help='hidden layer')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--drop-out', type=float, default=0.2, metavar='DR_R',
                    help='drop out rate')
parser.add_argument('--random-seed', type=float, default=0, metavar='RAND_SEED',
                    help='drop out rate')
args, unknown = parser.parse_known_args()
#%%
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

writer = SummaryWriter(comment="single_lstm")

SAVE_DATA_PATH = "./save_data"
writer.add_hparams({"epoch":N_EPOCH, "batch_size":BATCH_SIZE,
                    "hid_size":HID_SIZE, "hid_layer":HID_LAYER,
                    "learning_rate":LR, "drop_out":DROP_OUT_RATE,
                    "embedding_dim":EMBEDDING_DIM, "random seed": MANUAL_SEED,
                    "data_source":SAVE_DATA_PATH}, {})

device = torch.device("cuda:0")

training_data = np.loadtxt(f"{SAVE_DATA_PATH}/traj_x.txt", dtype=int)
training_label = np.loadtxt(f"{SAVE_DATA_PATH}/traj_y.txt", dtype=int)
training_label = training_label.reshape(-1, 1) #TODO check here
training_dataset = TrajDataset(training_data, training_label)
label_unique = np.loadtxt(f"{SAVE_DATA_PATH}/str_code.txt", dtype="U8")
num_class = len(label_unique)

training_dataloader = torch.utils.data.DataLoader(
                    training_dataset, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=4,
                    drop_last=False)

ix = np.arange(label_unique.size)
ix2words = dict(zip(ix, label_unique))
words2ix = dict(zip(label_unique, ix))

predict_dim = training_label.shape[1]

gru = GRU_NN(
    input_size= num_class,
    hid_size=HID_SIZE,
    hid_layers=HID_LAYER,
    output_size=num_class,
    num_class=num_class,
    drop_out_rate=DROP_OUT_RATE,
    embeded_size=EMBEDDING_DIM
)

torchsummaryX.summary(gru, torch.zeros((1, 1), dtype=torch.long))

criterion = torch.nn.NLLLoss(reduction="mean")
optimizer = torch.optim.Adam(gru.parameters(), lr=LR)

log_dir = writer.log_dir

total_step = 0

gru.to(device)


#%%
gru.train()
total_samples = training_dataloader.dataset.__len__()

for _epoc in range(N_EPOCH):
    for i_batch, batch_data in enumerate(training_dataloader):
        data_x, data_y = batch_data
        data_x = torch.transpose(data_x, 1, 0)

        data_x = data_x.to(device)
        data_y = data_y.to(device)

        data_predict, _ = gru(data_x)
        data_predict = data_predict[-predict_dim:, :] # here not right
        data_predict = data_predict.reshape(-1, num_class)
        data_y = data_y.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(data_predict, data_y)
        loss.backward()
        optimizer.step()

        if i_batch % 5 == 0:
            print(f"epoch {_epoc} ",
                  f"{i_batch*training_dataloader.batch_size/total_samples*100:.2f}% ",
                  f"{loss}")

        writer.add_scalar("loss/training_loss", loss, total_step)
        total_step += 1

    torch.save(gru.state_dict(), f"{log_dir}/rnn_model.pth")

gru.eval()
gru.to("cpu")

#%%
print("evaluating.........")

begin_code = torch.tensor([words2ix["00000000"]])
begin_code = begin_code.reshape(1,1)
begin_code = begin_code.repeat(5, 24)

fig_ge, fig_ge_3d = evaluation(gru, 60, begin_code, 6, 4, 5, ix2words)
writer.add_figure("generate_fig", fig_ge)
writer.add_figure("generate_fig_3d", fig_ge_3d)

writer.close()

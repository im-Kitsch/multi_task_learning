#%%

import torch


# need to firstly flatten label.
def one_hot_transfer(label, class_num):
    label = label.reshape(-1, 1)
    batch_len, _ = label.shape
    m_zeros = torch.zeros(batch_len, class_num)
    one_hot = m_zeros.scatter_(1, label, 1)  # (dim,index,value)

    return one_hot


class GRU_NN(torch.nn.Module):
    def __init__(self, input_size, hid_size, hid_layers, output_size, drop_out_rate):
        super(GRU_NN, self).__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hid_size,
            num_layers=hid_layers,
            bias=True,
            dropout=drop_out_rate
        )
        self.nn = torch.nn.Linear(hid_size, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
        return

    def forward(self, input_data):
        output, h_n = self.gru(input_data)
        output = self.nn(output)
        output = self.log_softmax(output)
        return output, h_n

def generate(model, traj_length, begin_code, num_class):

    model.eval()
    begin_code = one_hot_transfer(begin_code.reshape(-1, 1), num_class)
    begin_code = begin_code.reshape(-1, 1, num_class)
    traj_gen = torch.zeros(traj_length)

    for i in range(traj_length):
        predict, _ = model(begin_code)
        next_idx = torch.argmax(predict[-1])
        traj_gen[i] = next_idx

        next_tensor = torch.tensor([next_idx])
        next_tensor = one_hot_transfer(next_tensor, num_class)
        next_tensor = next_tensor.reshape(1, 1, -1)
        begin_code = torch.cat([begin_code, next_tensor], dim=0)
        begin_code = begin_code[1:]

    return traj_gen

gru = GRU_NN(
    input_size= 2792,
    hid_size=80,
    hid_layers=2,
    output_size=2792,
    drop_out_rate=0.2
)
gru.load_state_dict(torch.load("runs/Feb13_13-54-02_ip-172-30-1-47single_lstm/rnn_model.pth"))

traj_gen = generate(gru, 15, torch.tensor([60, 60, 60, 60, 60]), 2792)
pass

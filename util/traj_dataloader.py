import torch

import numpy as np
import pandas as pd

from operator import itemgetter


class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        super(TrajDataset, self).__init__()
        self.data = data
        self.label = label
        self.transform = transform
        return

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_x, data_y = self.data[idx, :], self.label[idx, :]
        return data_x, data_y


class PflowLoader:
    def __init__(self, data_path, dict_path, begin_ix, time_interval, num_per_day,
                 mode="normal", require="traj_code"):
        code = pd.read_csv(data_path, header=None, usecols=[14])
        code_arr = code.loc[:, 14].to_numpy(dtype="U8")
        del code

        code_arr = code_arr.reshape(-1, 1440)

        ix = begin_ix + np.arange(num_per_day) * time_interval
        code_arr = code_arr[:, ix]

        uniq_code = np.unique(code_arr)
        uniq_ix = np.arange(len(uniq_code))

        self.code2ix = dict(zip(uniq_code, uniq_ix))
        self.ix2code = dict(zip(uniq_ix, uniq_code))

        self.dataset = self.trans_code2ix(code_arr)
        del code_arr

        all_unique_code = np.loadtxt(f"{dict_path}/unique_code.txt", dtype="U8")
        all_unique_coordinate = np.loadtxt(f"{dict_path}/unique_coordinate.txt")

        self.code2lng = dict(zip(all_unique_code, all_unique_coordinate[:, 0]))
        self.code2lat = dict(zip(all_unique_code, all_unique_coordinate[:, 1]))

        return

    def trans_code2ix(self, code):
        orig_shape = code.shape
        code = code.reshape(-1)
        code = itemgetter(*code.tolist())(self.code2ix)
        code = np.array(code, dtype=int)
        code = code.reshape(orig_shape)
        return code

    def trans_ix2code(self, ix):
        orig_shape = ix.shape
        ix = ix.reshape(-1)
        ix = itemgetter(*ix.tolist())(self.ix2code)
        ix = np.array(ix, dtype="U8")
        ix = ix.reshape(orig_shape)
        return ix

    def trans_ix2coordi(self, ix):
        code = self.trans_ix2code(ix)
        orig_shape = code.shape
        code = code.reshape(-1)
        lng = itemgetter(*code.tolist())(self.code2lng)
        lat = itemgetter(*code.tolist())(self.code2lat)
        lng, lat = np.array(lng), np.array(lat)
        lng, lat = lng.reshape(orig_shape), lat.reshape(orig_shape)
        lng, lat = np.expand_dims(lng, -1), np.expand_dims(lat, -1)
        coordi = np.concatenate([lng, lat], axis=-1)
        return coordi


if __name__ == "__main__":

    traj_dataset = TrajDataset(np.random.random((9, 3)), np.random.random((9, 1)))

    dataloader = torch.utils.data.DataLoader(traj_dataset, batch_size=4,
                            shuffle=True, num_workers=4, drop_last=False)

    for i_batch, batch_data in enumerate(dataloader):
        batch_x, batch_y = batch_data
        print(batch_x)

    pflow_loader = PflowLoader("~/data/pflow_mini_preprocessed.csv", "dict_file/", 480, 15, 4)
    pflow_loader.trans_ix2code(pflow_loader.dataset)
    pflow_loader.trans_ix2coordi(pflow_loader.dataset)


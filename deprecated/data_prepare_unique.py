#%%
import pandas as pd

from util.coordi_map import Coordinate2MeshCodePandas
import numpy as np

from operator import itemgetter
#%%


def add_start_sig(start_sig, num, arr_in):
    start_arr = np.zeros((arr_in.shape[0], num), dtype=int)
    start_arr += start_sig
    return np.concatenate([start_arr, arr_in], axis=1)

#%%


DATA_PATH = "~/data"

df = pd.read_csv(f"{DATA_PATH}/pflow_mini_original.csv", header=None)#, nrows=10000)


#%%
_metric = 0  # drop metric

minutes_interval = 30
traj_n_samples = 24

traj_begin_index = 8 * 60
traj_end_index = traj_begin_index + traj_n_samples * minutes_interval

df_time = pd.to_datetime(df[3])
begin_df = df[(df_time.dt.hour == 0) & (df_time.dt.minute == 0)]

# last sample drop
begin_indices = begin_df.index
# filter repeated begin index
begin_indices = begin_indices[1:][begin_indices[1:] - begin_indices[0:-1] >= 24*60]
begin_indices = begin_indices.insert(0, 0)

n_samples = begin_indices.shape[0] - 1

trajectories = np.zeros((n_samples, traj_n_samples*minutes_interval), dtype="U8")

#%%
for i in range(n_samples):
    samp_index = begin_indices[i]
    sub_df = df[samp_index+traj_begin_index:samp_index+traj_end_index:1]
    sub_lng, sub_lat = sub_df.loc[:, 4], sub_df.loc[:, 5]
    sub_meshcode = Coordinate2MeshCodePandas(sub_lng, sub_lat)
    sub_meshcode = sub_meshcode.to_numpy().astype("U8")
    trajectories[i] = sub_meshcode

    print(f"{i*100/n_samples:.2f}% finished, {i} / {n_samples}")


#%%
sub_traj = np.zeros((n_samples, traj_n_samples), dtype="U8")

last_code = trajectories[:, [0]]
sub_traj[:, 0] = last_code.flatten()
for i in range(1, traj_n_samples):
    traj_interval = trajectories[:, i*minutes_interval : (i+1)*minutes_interval]
    next_idx = np.argmax(traj_interval != last_code, axis=1)
    next_idx = next_idx.reshape(-1, 1)
    next_traj = np.take_along_axis(traj_interval, next_idx, axis=1)
    last_code = next_traj
    sub_traj[:, i] = next_traj.flatten()


print(f">={_metric} unique code")
print(np.sum(np.sum(sub_traj[:, 1:] != sub_traj[:, :-1], axis =1) >= _metric))

print(np.sum(np.sum(trajectories[:, 1:] != trajectories[:, :-1], axis =1) >= _metric))

sub_traj_cut_short = sub_traj[np.sum(sub_traj[:, 1:] != sub_traj[:, :-1], axis =1) >= _metric]

#%%
words = np.unique(sub_traj_cut_short)
start_str = "00000000"
words = np.append(words, "00000000")
ix = np.arange(words.size)
ix2words = dict(zip(ix, words))
words2ix = dict(zip(words, ix))

n_row, tim_len = sub_traj_cut_short.shape
sub_traj_cut_short = sub_traj_cut_short.flatten()

traj_code = itemgetter(*sub_traj_cut_short.tolist())(words2ix)
traj_code = np.array(traj_code)
traj_code = traj_code.reshape(n_row, tim_len)
sub_traj_cut_short = sub_traj_cut_short.reshape(n_row, tim_len)

traj_code = add_start_sig(words2ix[start_str], 1, traj_code)

traj_x = traj_code[:, :-1]
traj_y = traj_code[:, 1:]

np.savetxt("./save_data/traj_x.txt", traj_x, fmt="%d")
np.savetxt("./save_data/traj_y.txt", traj_y, fmt="%d")
np.savetxt("./save_data/str_code.txt", words, fmt="%s")

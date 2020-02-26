#%%
import pandas as pd

from util.coordi_map import Coordinate2MeshCodePandas
import numpy as np

from operator import itemgetter

DATA_PATH = "~/data"
minutes_interval = 10
traj_n_samples = 10 * 6

traj_begin_index = 8 * 60
traj_end_index = traj_begin_index + traj_n_samples * minutes_interval

df = pd.read_csv(f"{DATA_PATH}/pflow_mini_original.csv", header=None)#, nrows=10000)


#%%

df_time = pd.to_datetime(df[3])
begin_df = df[(df_time.dt.hour == 0) & (df_time.dt.minute == 0)]

# last sample drop
n_samples = begin_df.shape[0] - 1
begin_indices = begin_df.index

trajectories = np.zeros((n_samples, traj_n_samples), dtype="U8")

#%%
for i in range(n_samples):
    samp_index = begin_indices[i]
    sub_df = df[samp_index+traj_begin_index:samp_index+traj_end_index:minutes_interval]
    sub_lng, sub_lat = sub_df.loc[:, 4], sub_df.loc[:, 5]
    sub_meshcode = Coordinate2MeshCodePandas(sub_lng, sub_lat)
    sub_meshcode = sub_meshcode.to_numpy().astype("U8")
    trajectories[i] = sub_meshcode

    print(f"{i*100/n_samples:.2f}% finished, {i} / {n_samples}")

#%%
unique_code = np.unique(trajectories)
start_str = "00000000"
unique_code = np.append(unique_code, start_str)
dict_code2str = dict(zip(np.arange(unique_code.shape[0]), unique_code))
dict_str2code = dict(zip(unique_code, np.arange(unique_code.shape[0])))
start_num = dict_str2code["00000000"]

#%%
n_row, tim_len = trajectories.shape
trajectories = trajectories.flatten()


traj_code = itemgetter(*trajectories.tolist())(dict_str2code)
traj_code = np.array(traj_code)
traj_code = traj_code.reshape(n_row, tim_len)
trajectories = trajectories.reshape(n_row, tim_len)

start_sig_len = 5
start_mat = np.zeros((n_row, start_sig_len), dtype=int)
start_mat += start_num

traj_extend = np.concatenate([start_mat, traj_code], axis=1)

#%%
indices_x = np.arange(tim_len)[:, None] + np.arange(start_sig_len)[None, :]

traj_x = traj_extend[:, indices_x]


#%%
traj_x = np.concatenate(traj_x, axis=0)
traj_y = traj_code.flatten().reshape(-1, 1)

np.savetxt("./save_data/traj_x.txt", traj_x, fmt="%d")
np.savetxt("./save_data/traj_y.txt", traj_y, fmt="%d")
np.savetxt("./save_data/str_code.txt", unique_code, fmt="%s")



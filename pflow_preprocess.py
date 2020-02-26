#%%
import pandas as pd

from util.coordi_map import Coordinate2MeshCodePandas, parse_MeshCode
import numpy as np

DATA_PATH = "~/data"
minutes_interval = 30
traj_n_samples = 10 * 2

traj_begin_index = 8 * 60
traj_end_index = traj_begin_index + traj_n_samples * minutes_interval

df = pd.read_csv(f"{DATA_PATH}/pflow_mini_original.csv", header=None)#, nrows=50000)

df_time = pd.to_datetime(df[3])

print("read data finished")
#%%
df_t1 = df[3][:-1]
df_t2 = df[3][1:]

t_mask = df_t1.to_numpy() != df_t2.to_numpy()
t_mask = np.append(t_mask, True)

n_samples = t_mask.size // 1440

#%%
new_df = df.loc[t_mask]
new_df = new_df.reset_index(drop=True)

n_samples = new_df.shape[0] // 1440

new_df = new_df.loc[:n_samples*1440-1]

del df
print("generate data finished")
#%%
sub_lng, sub_lat = new_df.loc[:, 4], new_df.loc[:, 5]
meshcode = Coordinate2MeshCodePandas(sub_lng, sub_lat)

filtered_record = pd.concat([new_df, meshcode], axis=1)
filtered_record.columns = range(filtered_record.shape[1])

print("generate code finished")

filtered_record.to_csv(f"{DATA_PATH}/pflow_mini_preprocessed.csv", index=False, header=False)

print("finished")

unique_meshcode = meshcode.unique()

del filtered_record, meshcode

#%%

num_unique = len(unique_meshcode)

unique_coordi = np.zeros((num_unique, 2))
for i in range(num_unique):
    lng, lat = parse_MeshCode(unique_meshcode[i])
    unique_coordi[i, 0], unique_coordi[i, 1] = lng, lat

np.savetxt("dict_file/unique_code.txt", unique_meshcode, fmt="%s")
np.savetxt("dict_file/unique_coordinate.txt", unique_coordi, fmt="%f")

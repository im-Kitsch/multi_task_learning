import torch
import seaborn as sns
import math
import numpy as np
import matplotlib.pyplot as plt


def random_choose_topk(logits, topk):
    assert logits.ndim == 3

    topk_val, topk_ix = torch.topk(logits, topk, dim=2)

    n_r, n_c = logits.shape[0], logits.shape[1]
    rand_ix = torch.randint(0, topk, (n_r, n_c, 1))

    choice_val = torch.gather(topk_val, 2, rand_ix)
    choice_ix = torch.gather(topk_ix, 2, rand_ix)
    choice_val, choice_ix = choice_val.squeeze(2), choice_ix.squeeze(2)
    return choice_val, choice_ix


def add_start_num_np(input_dt, start_num, repeat_time, batch_first):
    if batch_first:
        batch_size, seq_len = input_dt.shape
        start_vec = np.zeros((batch_size, repeat_time), dtype=int) + start_num
        input_dt = np.concatenate([start_vec, input_dt], axis=1)
    else:
        seq_len, batch_size = input_dt.shape
        start_vec = np.zeros((repeat_time, batch_size), dtype=int) + start_num
        input_dt = np.concatenate([start_vec, input_dt], axis=0)
    return input_dt


def eval_relativity(model, ground_truth, use_len, count_interval, num_class, batch_first):
    assert batch_first is False
    batch_size = ground_truth.shape[1]

    begin_code = ground_truth[:use_len]
    pred_len = ground_truth.shape[0] - use_len
    begin_code = torch.tensor(begin_code)
    gen_traj = model.generate_traj(begin_code, pred_len, batch_first=False)

    gen_traj = torch.cat([begin_code, gen_traj], dim=0)

    reference_ix = np.arange(num_class).reshape(1, 1, 1, -1)
    gen_traj = gen_traj.numpy()
    gen_traj = gen_traj.reshape(-1, count_interval, batch_size, 1)
    ground_truth = ground_truth.reshape(-1, count_interval, batch_size, 1)

    gen_traj_stats = np.sum(gen_traj == reference_ix, axis=(1, 2))
    ground_truth_stats = np.sum(ground_truth == reference_ix, axis=(1, 2))

    num_plots = gen_traj_stats.shape[0]
    n_c = 3
    n_r = math.ceil(num_plots/n_c)
    fig, axes = plt.subplots(nrows=n_r, ncols=n_c)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        sns.regplot(x=gen_traj_stats[i], y=ground_truth_stats[i], ax=ax)
    return fig

# generate_training_data.py
# Copyright 2022 Google LLC
# Copyright (c) 2020 Zonghan Wu

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# 数据加载和预处理脚本
""" Generating the data from disk files """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

import os
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    if len(df.shape) == 2:
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
    else:
        num_samples, num_nodes, dims = df.shape
        data = df


    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...].astype(np.float32)
        y_t = data[t + y_offsets, ...].astype(np.float32)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    if args.ds_name == "metr-la":
        df = pd.read_hdf(args.dataset_filename)
    else:
        df = pd.read_csv(args.dataset_filename, delimiter = ",", header=None)
        if args.ds_name == "traffic":
            df = df * 1000
        if args.ds_name == "ECG":
            df = df * 10

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    if args.ds_name == "metr-la":
        add_time_in_day = True
    else:
        add_time_in_day = False
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=add_time_in_day,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    if args.ds_name.lower() not in args.output_dir.lower():
        raise Exception("Incorrect output directory")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_name", type=str, default="metr-la", help="dataset name."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--dataset_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw dataset readings.",
    )
    args = parser.parse_args()
    main(args)


def generate_non_structural_data(ds_name, output_dir, dataset_filename, seq_len=12, pred_len=12):
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始数据
    print(f"Loading data from {dataset_filename}...")
    raw_data = np.loadtxt(dataset_filename, delimiter=",")
    print(f"Raw data shape: {raw_data.shape}")

    # 归一化处理
    scaler = StandardScaler()
    raw_data = scaler.fit_transform(raw_data)

    # 生成序列数据
    num_samples = raw_data.shape[0] - seq_len - pred_len + 1
    data = []
    target = []

    for i in range(num_samples):
        data.append(raw_data[i : i + seq_len, :])
        target.append(raw_data[i + seq_len : i + seq_len + pred_len, :])

    # 转换为numpy数组
    data = np.array(data)
    target = np.array(target)
    print(f"Processed data shape: {data.shape}, target shape: {target.shape}")

    # 数据拆分
    num_train = int(0.7 * len(data))
    num_val = int(0.15 * len(data))
    num_test = len(data) - num_train - num_val

    # 保存数据
    np.save(os.path.join(output_dir, "train_data.npy"), data[:num_train])
    np.save(os.path.join(output_dir, "val_data.npy"), data[num_train:num_train + num_val])
    np.save(os.path.join(output_dir, "test_data.npy"), data[num_train + num_val:])

    np.save(os.path.join(output_dir, "train_target.npy"), target[:num_train])
    np.save(os.path.join(output_dir, "val_target.npy"), target[num_train:num_train + num_val])
    np.save(os.path.join(output_dir, "test_target.npy"), target[num_train + num_val:])

    print(f"Data saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Non-Structural VSF Dataset")
    parser.add_argument("--ds_name", type=str, default="solar", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./data/solar_non_structural", help="Output directory")
    parser.add_argument("--dataset_filename", type=str, required=True, help="Input raw dataset file")
    args = parser.parse_args()

    generate_non_structural_data(args.ds_name, args.output_dir, args.dataset_filename)
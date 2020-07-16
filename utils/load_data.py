from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def get_scalar(files):
    """
    Get the MinMax data scaler from all the training data prices. Going through on its own loop to only have one
    copy of each price value, instead of doing it in the load_data function where windows cause values to be repeated

    :param files: Iterable of datafile locations
    :return: Fit scalar
    """
    data = np.array([])
    for path in files:
        df = pd.read_csv(path)
        data = np.concatenate((data, df["Close"].values))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data.reshape(-1, 1))

    return scaler


def load_data(data_dir, rolling_mean, input_length, output_length, scaler=None, single_file=None):
    data_files = Path(data_dir).glob("*.csv")
    if scaler is None:
        scaler = get_scalar(data_files)
        # re-initialize the generator, which gets used in the getting the scaler
        data_files = Path(data_dir).glob("*.csv")

    if single_file is not None:
        data_files = [Path(single_file)]

    return load_from_list(data_files, rolling_mean, input_length, output_length, scaler=scaler)


def load_from_list(data_list, rolling_mean, input_length, output_length, scaler=None):
    tickers = []
    data = None
    labels = None
    for path in data_list:
        ticker = [path.stem.split("_")[0]]

        df = pd.read_csv(path)
        price = scaler.transform(df["Close"].values.reshape(-1, 1)).ravel()

        windows = rolling_window(price, rolling_mean)
        rolled = windows.mean(axis=1)

        mean_windows = rolling_window(rolled, input_length + output_length)
        inpt = mean_windows[:, :input_length]
        output = mean_windows[:, input_length:]

        tickers += ticker * len(inpt)

        try:
            data = np.vstack((data, inpt))
            labels = np.vstack((labels, output))
        except ValueError:
            data = inpt
            labels = output

    return data[:, :, np.newaxis], labels, scaler, tickers

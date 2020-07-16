import numpy as np
import torch
from torch.utils import data

from utils.load_data import load_data


class StockData(data.Dataset):

    def __init__(self, data_dir, rolling_mean, input_length, output_length, scaler=None, single_file=None):
        self.data, self.label, self.scaler, self.tickers = load_data(data_dir,
                                                                     rolling_mean,
                                                                     input_length,
                                                                     output_length,
                                                                     scaler=scaler,
                                                                     single_file=single_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_cur = torch.from_numpy(self.data[item].astype(np.float32))
        label_cur = self.label[item]
        ticker_cur = self.tickers[item]
        return data_cur, label_cur, ticker_cur

    def get_scaler(self):
        return self.scaler

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy


class AudioFilterDataset(Dataset):

    def __init__(
        self,
        inputs: list[np.ndarray],
        fc_to_gen: int,
        fc_min: float,
        fc_max: float,
        make_filter_coef: callable,
        normalize_fc: callable,
    ):

        self.inputs = inputs[:]  # Stack into (n_sequences, 2, input_sizes)
        self.min_fc = fc_min
        self.max_fc = fc_max
        self.fc_to_gen = fc_to_gen
        self.make_filter_coefs = make_filter_coef
        self.normalize_fc = normalize_fc

    def __len__(self):
        return self.fc_to_gen

    def __getitem__(self, idx):
        fc = np.exp(np.random.uniform(np.log(self.min_fc), np.log(self.max_fc)))
        b, a = self.make_filter_coefs(fc)
        N = len(self.inputs)
        raw_signal = self.inputs[idx % N]  # preloaded dry signal
        target = scipy.signal.lfilter(b, a, raw_signal).astype(np.float32)
        fc_buffer = np.full(
            (len(raw_signal), 1), self.normalize_fc(fc), dtype=np.float32
        )
        x = np.concatenate([raw_signal[..., np.newaxis], fc_buffer], axis=-1)
        return torch.from_numpy(x), torch.from_numpy(target.reshape(len(target), 1))

import torch
from torch.utils.data import Dataset
import numpy as np


class AudioFilterDataset(Dataset):
    def __init__(
        self, inputs: list[np.ndarray], filtered: list[np.ndarray], buffer_size: int
    ):
        ins, targets = [], []

        for raw, filt in zip(inputs, filtered):
            fc_norm = raw[-1]
            in_size = len(raw) - 1

            raw_in = raw[:in_size]

            fc_channel = np.full((in_size, 1), fc_norm)
            inp = np.concatenate([raw_in[..., np.newaxis], fc_channel], axis=-1)

            ins.append(inp)
            targets.append(filt.reshape(len(filt), 1))

        self.inputs = ins  # Stack into (n_sequences, 2, input_sizes)
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

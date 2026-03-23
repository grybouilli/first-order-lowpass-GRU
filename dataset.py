import torch
from torch.utils.data import Dataset
import numpy as np


class AudioFilterDataset(Dataset):
    def __init__(
        self,
        samples: list[np.ndarray],
        filtered: list[np.ndarray],
        buffer_size: int,
        sample_rate: int,
    ):
        self.buffer_size = buffer_size
        inputs, targets = [], []

        for raw, filt in zip(samples, filtered):
            fc_norm = raw[-1]
            n_buffers = (len(raw) - 1) // buffer_size

            raw_buffers = raw[: n_buffers * buffer_size].reshape(n_buffers, buffer_size)
            filt_buffers = filt[: n_buffers * buffer_size].reshape(
                n_buffers, buffer_size
            )

            fc_channel = np.full((n_buffers, buffer_size, 1), fc_norm)
            inp = np.concatenate([raw_buffers[..., np.newaxis], fc_channel], axis=-1)
            tgt = filt_buffers[..., np.newaxis]

            inputs.append(inp)
            targets.append(tgt)

        # Stack into (n_sequences, n_buffers, buffer_size, 2/1)
        self.inputs = torch.tensor(np.stack(inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.stack(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

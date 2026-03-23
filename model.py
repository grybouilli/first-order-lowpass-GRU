from torch import nn, Tensor


class LowpassRNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, conditioned: bool = True):
        super(LowpassRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conditioned = conditioned

        input_size = 2 if conditioned else 1  # [sample, fc_norm] or just [sample]

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # expects (batch, seq_len, input_size)
        )
        self.fc = nn.Linear(hidden_size, 1)  # one output sample per timestep

    def forward(self, x: Tensor, hidden: Tensor | None = None):
        """
        Args:
            x:      (batch_size, buffer_size, input_size)  — one timestep per sample
            hidden: (num_layers, batch_size, hidden_size)  — carried across buffers
        Returns:
            output: (batch_size, buffer_size, 1)
            hidden: (num_layers, batch_size, hidden_size)  — to be passed to next buffer
        """
        gru_out, hidden = self.gru(
            x, hidden
        )  # gru_out: (batch, buffer_size, hidden_size)
        output = self.fc(gru_out)  # (batch, buffer_size, 1)
        return output, hidden

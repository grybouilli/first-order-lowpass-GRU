from create_dataset import normalize_freq
from model import LowpassRNN
import numpy as np
import torch


def run_inference(
    model: LowpassRNN,
    input: np.ndarray,
    fc_norm: float,
    buffer_size: int,
) -> np.ndarray:
    """
    Args:
        model:       trained LowpassRNN
        input:       raw audio signal of length N, where N is a multiple of buffer_size
        fc_norm:     normalized cutoff frequency (2 * cutoff_freq / sample_rate)
        buffer_size: must match the buffer_size used during training
        sample_rate: must match the sample_rate used during training
    Returns:
        filtered signal of length N
    """
    model.eval()

    n_buffers = len(input) // buffer_size
    output_buffers = []
    hidden = None

    with torch.no_grad():
        for i in range(n_buffers):
            buffer = input[i * buffer_size : (i + 1) * buffer_size]
            x = torch.from_numpy(buffer).float()  # (buffer_size,)
            fc_channel = torch.full((buffer_size,), fc_norm)
            x = torch.stack([x, fc_channel], dim=-1).unsqueeze(0)  # (1, buffer_size, 2)
            output, hidden = model(x, hidden)

            # .cpu() handles both CUDA and CPU tensors safely before .numpy()
            output_buffers.append(output.squeeze().cpu().numpy())  # (buffer_size,)

    return np.concatenate(output_buffers)  # (N,)


def load_gru_model(
    filename: str, hidden_size: int, num_layers: int, device: str = "cpu"
):
    try:
        checkpoint = torch.load(filename)
    except RuntimeError:
        checkpoint = torch.load(filename, map_location=torch.device(device))
    gru = LowpassRNN(hidden_size=hidden_size, num_layers=num_layers)

    try:
        gru.load_state_dict(checkpoint)
    except Exception:
        # The model was compiled during training, so the ckpt keys are different
        try:
            state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            gru.load_state_dict(state_dict)
        except Exception:
            # This is a final model and not a ckpt
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
            gru.load_state_dict(state_dict)

    return gru

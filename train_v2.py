import model
import dataset_v2
import os, argparse
import soundfile as sf
import numpy as np
import torchaudio
from torch.utils.data import DataLoader
from torch import Tensor, tensor, save, where, full_like, ones_like
from torch.optim import Adam
from torch import nn, fft
from pathlib import Path
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument(
    "--buffer_size",
    type=int,
    default=1024,
    help="Amount of samples passed as input to the GRU model during forward pass",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="./dataset-0",
    help="Folder which should contain two subfolders ./inputs and ./expected, that represent the training dataset generated with create_dataset.py",
)

parser.add_argument(
    "--val_dataset",
    type=str,
    default="./dataset-1",
    help="Folder which should contain two subfolders ./inputs and ./expected, that represent the validation dataset generated with create_dataset.py",
)
parser.add_argument(
    "--epochs", type=int, default=50, help="The amount of epoch for training"
)
parser.add_argument(
    "--hidden_size", type=int, default=64, help="The hidden size of the GRU"
)
parser.add_argument(
    "--num_layers", type=int, default=2, help="The number of layers in the GRU"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="The batch size in dataset (default is 8)",
)

parser.add_argument(
    "--initial_lr",
    type=float,
    default=10**-3,
    help="The initial learning rate (default is 10**-3)",
)

parser.add_argument(
    "--notes",
    type=str,
    default="",
    help="Notes about the training",
)

args = parser.parse_args()
train_sample_folder = args.dataset
val_sample_folder = args.val_dataset

train_inputs = []
train_outputs = []
val_inputs = []
val_outputs = []

if not os.path.exists(train_sample_folder):
    raise FileNotFoundError(f"Training dataset {train_sample_folder} does not exist")
if not os.path.exists(val_sample_folder):
    raise FileNotFoundError(f"Training dataset {val_sample_folder} does not exist")


def load_data_from_folder(path_to_dataset: str, inputs: list, outputs: list):
    xpath = os.path.join(path_to_dataset, "inputs")
    ypath = os.path.join(path_to_dataset, "expected")

    prev_sz = 0
    for filename in sorted(os.listdir(xpath)):
        data = np.load(os.path.join(xpath, filename))
        inputs.append(data)
        sz = len(data)
        if prev_sz != 0 and prev_sz != sz:
            print("different size in inputs : {} != {}".format(prev_sz, sz))
            print(f"file {path_to_dataset} {filename} has size {sz}")
            # raise Exception()
        prev_sz = sz

    for filename in sorted(os.listdir(ypath)):
        data = np.load(os.path.join(ypath, filename))
        outputs.append(data)


load_data_from_folder(train_sample_folder, train_inputs, train_outputs)
load_data_from_folder(val_sample_folder, val_inputs, val_outputs)

print(f"Loaded {len(train_inputs)} train sequences")
print(f"Loaded {len(val_inputs)} val sequences")
print(f"Input sequence length:  {len(train_inputs[0])} samples")
print(f"Output sequence length: {len(train_inputs[0])} samples")

train_ds = dataset_v2.AudioFilterDataset(train_inputs, train_outputs, args.buffer_size)
val_ds = dataset_v2.AudioFilterDataset(val_inputs, val_outputs, args.buffer_size)

train_dataloader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True
)
val_dataloader = DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True
)

checkpoint_folder_base = f"checkpoints-{args.dataset.strip(".").strip("/")}"
checkpoint_folder = ""
run_id = 0

while True:
    try:
        checkpoint_folder = f"{checkpoint_folder_base}-{run_id}"
        os.makedirs(checkpoint_folder)
        break
    except OSError:
        if Path(checkpoint_folder).is_dir():
            run_id = run_id + 1
            continue
        raise
print(f"saving model to {checkpoint_folder}")

best_loss = float("inf")

import torch

device = None
cuda_available = torch.cuda.is_available()
if cuda_available:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

gru = model.LowpassRNN(hidden_size=args.hidden_size, num_layers=args.num_layers).to(
    device
)
gru = torch.compile(gru)
optimizer = Adam(gru.parameters(), lr=args.initial_lr)
criterion = nn.MSELoss()


def spectral_loss(output: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    # output, target: (batch, buffer_size, 1)
    O = torch.fft.rfft(output.squeeze(-1), dim=-1)
    T = torch.fft.rfft(target.squeeze(-1), dim=-1)
    log_O = torch.log(torch.abs(O) + eps)
    log_T = torch.log(torch.abs(T) + eps)
    return nn.functional.mse_loss(log_O, log_T)


# reconstruction loss used in DDSP paper
nfft = int(args.buffer_size * 0.5)
spectrogram = torchaudio.transforms.Spectrogram(n_fft=nfft).to(device=device)


def multi_scale_spectral_loss(
    output: Tensor,
    target: Tensor,
    eps: float = 1e-7,
    alpha: float = 1.0,
) -> Tensor:
    S_O = spectrogram(output.squeeze(-1))
    S_T = spectrogram(target.squeeze(-1))

    return torch.sum(torch.abs(S_O - S_T)) + alpha * torch.sum(
        torch.abs(torch.log(S_O + eps) - torch.log(S_T + eps))
    )


def batch_loop(batch_inputs, batch_targets, opt: Adam, train: bool) -> float:
    # batch_inputs: (batch, seq_len, 1) — move whole batch at once
    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)

    if train:
        opt.zero_grad()

    B, total_len, C = batch_inputs.shape
    buffers = total_len // args.buffer_size
    hidden = None
    loss_accum = torch.tensor(0.0, device=device)

    for buffer in range(buffers):
        beg = buffer * args.buffer_size
        end = (buffer + 1) * args.buffer_size

        x = batch_inputs[:, beg:end, :]  # (B, buffer_size, 1)
        target = batch_targets[:, beg:end, :]  # (B, buffer_size, 1)

        y_pred, hidden = gru(x, hidden)
        hidden = hidden.detach()
        loss_accum += multi_scale_spectral_loss(y_pred, target)

    loss_accum /= buffers

    if train:
        loss_accum.backward()
        torch.nn.utils.clip_grad_norm_(gru.parameters(), 1.0)
        opt.step()

    return loss_accum.item()


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3
)


def progress_bar(epoch: int, total_epoch: int, current: int, total: int, width=40):
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(
        f"\r[{bar}] {percent:.0%} ({current}/{total} Batch) | Epoch {epoch+1}/{total_epoch}",
        end="",
        flush=True,
    )


train_batch_amount = len(train_dataloader)
valid_batch_amount = len(val_dataloader)
for epoch in range(args.epochs):
    print(f"\nStarting Epoch {epoch+1}")
    epoch_loss = 0.0
    validation_loss = 0.0
    # Training part
    gru.train()
    for batch_idx, (batch_inputs, batch_targets) in enumerate(train_dataloader):
        progress_bar(epoch, args.epochs, batch_idx + 1, train_batch_amount)
        epoch_loss += batch_loop(batch_inputs, batch_targets, optimizer, True)

    # Validation part
    print("\nValidation set")
    gru.eval()
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_targets) in enumerate(val_dataloader):
            progress_bar(epoch, args.epochs, batch_idx + 1, valid_batch_amount)
            validation_loss += batch_loop(batch_inputs, batch_targets, optimizer, False)

    avg_train_loss = epoch_loss / train_batch_amount
    avg_valid_loss = validation_loss / valid_batch_amount
    # Save checkpoint every epoch
    print(
        f"\n── Epoch {epoch+1} complete | Avg train loss: {avg_train_loss:.6f} | Avg validation loss: {avg_valid_loss:.6f}"
    )
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": gru.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "valid_loss": avg_valid_loss,
        "buffer_size": args.buffer_size,
        "initial_lr": args.initial_lr,
        "batch_size": args.batch_size,
        "notes": args.notes,
    }
    save(checkpoint, os.path.join(checkpoint_folder, f"checkpoint_epoch{epoch+1}.pt"))

    # Save best model separately
    if avg_valid_loss < best_loss:
        best_loss = avg_valid_loss
        save(checkpoint, os.path.join(checkpoint_folder, "best.pt"))
        print(f"   ↳ New best model saved (loss: {best_loss:.6f})")
    if cuda_available:
        torch.cuda.empty_cache()

    scheduler.step(avg_valid_loss)

save(gru.state_dict(), os.path.join(checkpoint_folder, "lowpass_rnn.pt"))
print("Final model saved to " + os.path.join(checkpoint_folder, "lowpass_rnn.pt"))

import model
import dataset
import os, argparse
import soundfile as sf
import numpy as np
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

train_ds = dataset.AudioFilterDataset(train_inputs, train_outputs, args.buffer_size)
val_ds = dataset.AudioFilterDataset(val_inputs, val_outputs, args.buffer_size)

train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

checkpoint_folder = "checkpoints"

run_id = 0

while True:
    try:
        checkpoint_folder = f"checkpoints-{run_id}"
        os.makedirs(checkpoint_folder)
        break
    except OSError:
        if Path(checkpoint_folder).is_dir():
            run_id = run_id + 1
            continue
        raise

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


def batch_loop(batch_inputs, batch_targets, opt: Adam, train: bool) -> Tensor:
    hidden = None

    if train:
        opt.zero_grad()
    batch_loss = tensor(0.0, device=device)

    all_outputs = []
    all_targets = []
    for t in range(batch_inputs.shape[1]):
        x = batch_inputs[:, t, :, :].to(device)
        y = batch_targets[:, t, :, :].to(device)
        output, hidden = gru(x, hidden)
        hidden = hidden.detach()
        mse = criterion(output, y)
        batch_loss = batch_loss + mse
        all_outputs.append(output)
        all_targets.append(y)

    full_output = torch.cat(all_outputs, dim=1)  # (batch, total_samples, 1)
    full_target = torch.cat(all_targets, dim=1)

    batch_loss = batch_loss + 0.0001 * spectral_loss(full_output, full_target)
    batch_loss /= batch_inputs.shape[1]
    if train:
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(gru.parameters(), max_norm=1.0)  # add this
        opt.step()

    return batch_loss


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
    epoch_loss = 0.0
    validation_loss = 0.0
    # Training part
    gru.train()
    for batch_idx, (batch_inputs, batch_targets) in enumerate(train_dataloader):
        progress_bar(epoch, args.epochs, batch_idx + 1, train_batch_amount)
        epoch_loss += batch_loop(batch_inputs, batch_targets, optimizer, True).item()

    # Validation part
    print("\nValidation set\n")
    gru.eval()
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_targets) in enumerate(val_dataloader):
            progress_bar(epoch, args.epochs, batch_idx + 1, valid_batch_amount)
            validation_loss += batch_loop(
                batch_inputs, batch_targets, optimizer, False
            ).item()

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

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
    help="Folder which should contain two subfolders ./inputs and ./expected, that represent the training dataset generated with create_dataset_v2.py",
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
    "--val_size",
    type=float,
    default=0.2,
    help="The ratio of validation to training set (default is 0.2)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="The batch size in dataset (default is 8)",
)

args = parser.parse_args()
sample_folder = args.dataset

xpath = os.path.join(sample_folder, "inputs")
ypath = os.path.join(sample_folder, "expected")

inputs = []
outputs = []
prev_sz = 0
for filename in sorted(os.listdir(xpath)):
    data = np.load(os.path.join(xpath, filename))
    inputs.append(data)
    sz = len(data)
    if prev_sz != 0 and prev_sz != sz:
        print("different size in inputs : {} != {}".format(prev_sz, sz))
    prev_sz = sz

for filename in sorted(os.listdir(ypath)):
    data = np.load(os.path.join(ypath, filename))
    outputs.append(data)

print(f"Loaded {len(inputs)} sequences")
print(f"Input sequence length:  {len(inputs[0])} samples")
print(f"Output sequence length: {len(outputs[0])} samples")

train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
    inputs, outputs, test_size=args.val_size, random_state=42
)

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

optimizer = Adam(gru.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(args.epochs):
    epoch_loss = 0.0
    validation_loss = 0.0
    # Training part
    gru.train()
    for batch_idx, (batch_inputs, batch_targets) in enumerate(train_dataloader):
        hidden = None
        optimizer.zero_grad()
        batch_loss = tensor(0.0)

        for t in range(batch_inputs.shape[1]):
            x = batch_inputs[:, t, :, :].to(device)
            y = batch_targets[:, t, :, :].to(device)
            output, hidden = gru(x, hidden)
            hidden = hidden.detach()
            batch_loss = batch_loss + criterion(output, y)
        batch_loss /= batch_inputs.shape[1]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
        print(
            f"Epoch {epoch+1}/{args.epochs} | Training Batch {batch_idx+1}/{len(train_dataloader)} | Training Batch Loss: {batch_loss.item():.6f}"
        )

    # Validation part
    gru.eval()
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_targets) in enumerate(val_dataloader):
            hidden = None
            batch_loss = tensor(0.0)

            for t in range(batch_inputs.shape[1]):
                x = batch_inputs[:, t, :, :].to(device)
                y = batch_targets[:, t, :, :].to(device)
                output, hidden = gru(x, hidden)
                hidden = hidden.detach()
                batch_loss = batch_loss + criterion(output, y)
            validation_loss += batch_loss.item() / batch_inputs.shape[1]

    avg_train_loss = epoch_loss / len(train_dataloader)
    avg_valid_loss = validation_loss / len(val_dataloader)
    # Save checkpoint every epoch
    print(
        f"── Epoch {epoch+1} complete | Avg train loss: {avg_train_loss:.6f} | Avg validation loss: {avg_valid_loss:.6f}"
    )
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": gru.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_train_loss,
    }
    save(checkpoint, os.path.join(checkpoint_folder, f"checkpoint_epoch{epoch+1}.pt"))

    # Save best model separately
    if avg_valid_loss < best_loss:
        best_loss = avg_valid_loss
        save(checkpoint, os.path.join(checkpoint_folder, "best.pt"))
        print(f"   ↳ New best model saved (loss: {best_loss:.6f})")
    if cuda_available:
        torch.cuda.empty_cache()

save(gru.state_dict(), os.path.join(checkpoint_folder, "lowpass_rnn.pt"))
print("Final model saved to " + os.path.join(checkpoint_folder, "lowpass_rnn.pt"))

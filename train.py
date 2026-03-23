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

parser = argparse.ArgumentParser()

parser.add_argument("--sample_rate", type=int, default=44100)
parser.add_argument("--buffer_size", type=int, default=1024)
parser.add_argument("--dataset", type=str, default="./dataset-0")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--hidden_size", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=1)

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

ds = dataset.AudioFilterDataset(inputs, outputs, args.buffer_size, args.sample_rate)

dataloader = DataLoader(ds, batch_size=8, shuffle=True)

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

    for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
        hidden = None
        optimizer.zero_grad()
        total_loss = tensor(0.0)

        for t in range(batch_inputs.shape[1]):
            x = batch_inputs[:, t, :, :].to(device)
            y = batch_targets[:, t, :, :].to(device)
            output, hidden = gru(x, hidden)
            # hidden = hidden.detach()
            total_loss = total_loss + criterion(output, y)

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        print(
            f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {total_loss.item():.6f}"
        )

    avg_loss = epoch_loss / len(dataloader)
    print(f"── Epoch {epoch+1} complete | Avg loss: {avg_loss:.6f}")

    # Save checkpoint every epoch
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": gru.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }
    save(checkpoint, os.path.join(checkpoint_folder, f"checkpoint_epoch{epoch+1}.pt"))

    # Save best model separately
    if avg_loss < best_loss:
        best_loss = avg_loss
        save(checkpoint, os.path.join(checkpoint_folder, "best.pt"))
        print(f"   ↳ New best model saved (loss: {best_loss:.6f})")
    if cuda_available:
        torch.cuda.empty_cache()

save(gru.state_dict(), os.path.join(checkpoint_folder, "lowpass_rnn.pt"))
print("Final model saved to " + os.path.join(checkpoint_folder, "lowpass_rnn.pt"))

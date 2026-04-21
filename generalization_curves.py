import os
import torch
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--folder", type=str, required=True, help="Path to checkpoint folder"
)
parser.add_argument(
    "--interval", type=int, default=5, help="Seconds between folder checks"
)
args = parser.parse_args()


def get_checkpoints(checkpoint_folder):
    """Parses the folder and returns sorted loss data."""
    checkpoints = []
    for filename in os.listdir(checkpoint_folder):
        if filename.endswith(".pt") and "checkpoint_epoch" in filename:
            path = os.path.join(checkpoint_folder, filename)
            try:
                # use weights_only=True for security if using newer torch versions
                cp = torch.load(path, map_location="cpu", weights_only=False)
                checkpoints.append(
                    {
                        "epoch": cp["epoch"],
                        "train_loss": cp["train_loss"],
                        "valid_loss": cp["valid_loss"],
                    }
                )
            except Exception as e:
                # Sometimes files are caught while being written
                print(f"Skipping {filename}: {e}")
                continue

    checkpoints.sort(key=lambda x: x["epoch"])
    return checkpoints


def live_plot(checkpoint_folder, interval):
    # Enable interactive mode
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    last_count = -1

    print(f"Monitoring {checkpoint_folder}... Press Ctrl+C to stop.")

    try:
        while True:
            # Check how many relevant files are in the folder
            current_files = [
                f for f in os.listdir(checkpoint_folder) if f.endswith(".pt")
            ]

            if len(current_files) != last_count:
                data = get_checkpoints(checkpoint_folder)

                if data:
                    epochs = [cp["epoch"] for cp in data]
                    train_loss = [cp["train_loss"] for cp in data]
                    valid_loss = [cp["valid_loss"] for cp in data]

                    ax.clear()
                    ax.plot(epochs, train_loss, marker="o", label="Train Loss")
                    ax.plot(epochs, valid_loss, marker="o", label="Validation Loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(
                        f"Training vs Validation Loss (Updated: {time.strftime('%H:%M:%S')})"
                    )
                    ax.legend()
                    ax.grid(True)

                    plt.draw()
                    plt.savefig("losses.png", dpi=150)

                    last_count = len(current_files)
                    print(
                        f"Plot updated at {time.strftime('%H:%M:%S')} with {last_count} checkpoints."
                    )

            # This is crucial: it handles the GUI events and sleeps for the interval
            plt.pause(interval)

    except KeyboardInterrupt:
        print("\nStopping monitor...")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    live_plot(args.folder, args.interval)

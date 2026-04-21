import torch
import model
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--buffer_size", type=int, default=96)
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--model", type=str, default="lowpass_rnn.pt")
parser.add_argument("--model_out", type=str, default="lowpass_rnn.onnx")
parser.add_argument(
    "--no_dynamic_shapes",
    default=False,
    help="If set, the model will be compiled with fixed input shapes",
    action="store_true",
)
args = parser.parse_args()

try:
    checkpoint = torch.load(args.model)
except RuntimeError:
    checkpoint = torch.load(args.model, map_location=torch.device("cpu"))

gru = model.LowpassRNN(hidden_size=args.hidden_size, num_layers=args.num_layers)

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
gru.eval()

buffer_size = args.buffer_size
hidden_size = args.hidden_size
num_layers = args.num_layers

import torch
import torch.onnx

gru.eval()

dummy_x = torch.randn(1, buffer_size, 2)
dummy_hidden = torch.zeros(2, 1, hidden_size)

dynamic_axes = None
if not args.no_dynamic_shapes:
    dynamic_axes = {
        "x": {1: "buffer_size"},
        "output": {1: "buffer_size"},
    }
print(dynamic_axes)
torch.onnx.export(
    gru,
    (dummy_x, dummy_hidden),
    args.model_out,
    input_names=["x", "hidden_in"],
    output_names=["output", "hidden_out"],
    dynamic_axes=dynamic_axes,
    dynamo=False,  # force legacy TorchScript-based exporter
)

# Immediately validate
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(args.model_out)
out_onnx, h_onnx = sess.run(
    None,
    {
        "x": dummy_x.numpy(),
        "hidden_in": dummy_hidden.numpy(),
    },
)

with torch.no_grad():
    out_pt, h_pt = gru(dummy_x, dummy_hidden)

print("ONNX out[0,0,0]:", out_onnx[0, 0, 0])
print("ONNX h[0,0,:4]:", h_onnx[0, 0, :4])

print("Max output diff:", np.abs(out_pt.numpy() - out_onnx).max())
print("Max hidden diff:", np.abs(h_pt.numpy() - h_onnx).max())

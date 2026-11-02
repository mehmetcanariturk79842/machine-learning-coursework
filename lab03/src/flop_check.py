import random
from models import MLP, CNN, AttentionModel
import numpy as np
import torch
import copy
from torch import nn
from thop import profile

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

linear = nn.Linear(784, 256)
linear_input = torch.zeros(1, 784)

linear_macs, _ = profile(linear, inputs=(linear_input,), verbose=False)
print("Linear MACs:", int(linear_macs))
print("Linear FLOPs (2 x MACs):", int(2 * linear_macs))

conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
conv_input = torch.zeros(1, 16, 14, 14)

conv_macs, _ = profile(conv, inputs=(conv_input,), verbose=False)
print("Conv MACs:", int(conv_macs))
print("Conv FLOPs (2 x MACs):", int(2 * conv_macs))

attention = AttentionModel()
attention.eval()

attention_input = torch.zeros(1, 1, 28, 28)

attention_macs, _ = profile(
    attention,
    inputs=(attention_input,),
    verbose=False,
)

print("Attention model MACs reported by thop:", int(attention_macs))
print("Attention model FLOPs (2 x MACs):", int(2 * attention_macs))

encoder = copy.deepcopy(attention.encoder)
encoder.eval()

encoder_input = torch.zeros(1, 16, 64)

encoder_macs, _ = profile(
    encoder,
    inputs=(encoder_input,),
    verbose=False,
)

print("Encoder MACs reported by thop:", int(encoder_macs))
print("Encoder FLOPs (2 x MACs):", int(2 * encoder_macs))
models = [
    ("MLP", MLP(), 469504),
    ("CNN", CNN(), 2063488),
    ("Attention", AttentionModel(), 2329856),
]

print("\nFull model comparison:")
for name, model, hand_flops in models:
    model.eval()
    sample = torch.zeros(1, 1, 28, 28)

    macs, _ = profile(model, inputs=(sample,), verbose=False)
    thop_flops = int(2 * macs)

    print(f"{name}: hand={hand_flops}, thop={thop_flops}")
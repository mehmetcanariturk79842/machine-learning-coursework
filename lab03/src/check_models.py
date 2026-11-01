import random
from torchinfo import summary
import numpy as np
import torch

from models import MLP, CNN, AttentionModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

model = AttentionModel()
model.eval()

image = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)

captured = []


def capture_patches(module, inputs):
    captured.append(inputs[0].detach().clone())


hook = model.projection.register_forward_pre_hook(capture_patches)

with torch.inference_mode():
    model(image)

hook.remove()

patches = captured[0]

print("Patch shape:", patches.shape)

all_match = True
for row in range(4):
    for col in range(4):
        expected = image[0, 0, row * 7:(row + 1) * 7, col * 7:(col + 1) * 7]
        actual = patches[0, row * 4 + col].reshape(7, 7)
        if not torch.equal(actual, expected):
            all_match = False

print("All 16 patches match:", all_match)

summaries = []

for name, model in [
    ("MLP", MLP()),
    ("CNN", CNN()),
    ("Attention", AttentionModel()),
]:
    model_summary = summary(
        model,
        input_size=(1, 1, 28, 28),
        verbose=0,
    )
    print(f"{name} parameters: {model_summary.total_params}")
    summaries.append(f"===== {name} =====\n{model_summary}\n")

with open("lab03/results/model_summaries.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(summaries))
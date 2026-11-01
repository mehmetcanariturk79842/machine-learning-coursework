import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("lab02/results/training_log.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, width in zip(axes, [16, 64]):
    for learning_rate in [0.001, 0.01, 0.1]:
        run = data[
            (data["width"] == width)
            & (data["learning_rate"] == learning_rate)
        ]

        ax.plot(run["epoch"], run["training_loss"], label=f"lr={learning_rate}")

    ax.set_title(f"Hidden width: {width}")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel("Training loss")
fig.tight_layout()
fig.savefig("lab02/results/loss_curves.png", dpi=150)

print("Saved: lab02/results/loss_curves.png")
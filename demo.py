import matplotlib.pyplot as plt
import seaborn as sns
from src.main import *

def visualize_attention(model, x):
    model.eval()

    with torch.no_grad():
        _, unmasked_weights = model(x, mask=None)
        _, masked_weights = model(x, mask=True)

    unmasked = unmasked_weights[0].cpu().numpy()
    masked = masked_weights[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, weights, title in zip(
        axes,
        [unmasked, masked],
        ["Unmasked Self-Attention", "Masked Self-Attention"]
    ):
        sns.heatmap(
            weights,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=True
        )

        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

    plt.tight_layout()
    plt.show()

model = SelfAttention(hidden_states=128)
x = torch.randn(2, 6, 128)
visualize_attention(model, x)
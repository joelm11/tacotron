"""Utility helpers for inspecting and visualising model internals."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def get_logs_dir() -> Path:
    """Return the project's ``logs/`` directory, creating it if needed."""
    from taco.data import get_project_root

    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def plot_attention(attn_weights, save_path, title=None):
    """Render an attention alignment map to ``save_path``.

    Args:
        attn_weights: Attention weights of shape ``(T, S)`` or ``(B, T, S)``,
            where ``T`` and ``S`` are the decoder and encoder sequence lengths.
            A torch tensor or numpy array is accepted; batched input plots
            item ``0``.
        save_path: File to write the rendered plot to.
        title: Optional title for the plot.

    Returns:
        The :class:`~pathlib.Path` the plot was written to.
    """
    if hasattr(attn_weights, "detach"):
        attn_weights = attn_weights.detach().cpu().numpy()
    attn_weights = np.asarray(attn_weights, dtype=float)

    if attn_weights.ndim == 3:
        attn_weights = attn_weights[0]
    if attn_weights.ndim != 2:
        raise ValueError(
            f"Expected attention weights with 2 or 3 dimensions, got shape {attn_weights.shape}"
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(
        attn_weights.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="viridis",
    )
    ax.set_xlabel("Decoder step")
    ax.set_ylabel("Encoder step")
    if title:
        ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

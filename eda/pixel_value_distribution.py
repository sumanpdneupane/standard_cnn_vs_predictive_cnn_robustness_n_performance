import numpy as np
import torch
from matplotlib import pyplot as plt


def get_pixels(dataset):
    all_pixels = []
    for image, _ in dataset:
        all_pixels.append(image.numpy().flatten())
    return np.concatenate(all_pixels)


def plot_pixel_histograms(datasets_info, save=False):
    fig, axes = plt.subplots(1, len(datasets_info), figsize=(18, 4))
    fig.suptitle("Section 4 — Pixel Value Distribution",
                 fontsize=13, fontweight='bold')

    for i, (dataset, name, color) in enumerate(datasets_info):
        pixels = get_pixels(dataset)
        mean   = pixels.mean()
        std    = pixels.std()

        axes[i].hist(pixels, bins=50, color=color,
                     edgecolor='black', linewidth=0.3, alpha=0.85)
        axes[i].axvline(mean, color='red', linestyle='--',
                        linewidth=1.5, label=f'Mean: {mean:.3f}')
        axes[i].axvline(mean + std, color='orange', linestyle=':',
                        linewidth=1.5, label=f'Std:  {std:.3f}')
        axes[i].axvline(mean - std, color='orange', linestyle=':',
                        linewidth=1.5)
        axes[i].set_title(f"4.{i+1}  {name}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Pixel Value", fontsize=9)
        axes[i].set_ylabel("Frequency",   fontsize=9)
        axes[i].legend(fontsize=8)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("section4_pixel_histogram.png",
                    dpi=150, bbox_inches='tight')
        print("Saved: section4_pixel_histogram.png")


def get_pixel_stats(datasets_info):
    data = []
    for dataset, name, _ in datasets_info:
        pixels   = get_pixels(dataset)
        channels = dataset[0][0].shape[0]

        if channels == 1:
            data.append([
                name, "Grayscale",
                f"{pixels.mean():.4f}",
                f"{pixels.std():.4f}",
                f"{pixels.min():.4f}",
                f"{pixels.max():.4f}"
            ])
        else:
            imgs = torch.stack([img for img, _ in dataset])
            for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                ch = imgs[:, ch_idx, :, :].numpy().flatten()
                data.append([
                    f"{name} ({ch_name})", "Color",
                    f"{ch.mean():.4f}",
                    f"{ch.std():.4f}",
                    f"{ch.min():.4f}",
                    f"{ch.max():.4f}"
                ])
    return data


def plot_pixel_stats_table(data, save=False):
    columns = ["Dataset", "Type", "Mean", "Std Dev", "Min", "Max"]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis('off')

    table = ax.table(
        cellText  = data,
        colLabels = columns,
        loc       = 'center',
        cellLoc   = 'center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for col in range(len(columns)):
        table[0, col].set_facecolor('#000000')
        table[0, col].set_text_props(color='white', fontweight='bold')

    row_colors = ['#f2f2f2', '#ffffff']
    for row in range(1, len(data) + 1):
        for col in range(len(columns)):
            table[row, col].set_facecolor(row_colors[row % 2])
        table[row, 0].set_text_props(fontweight='bold')
        table[row, 0].set_facecolor('#d0e8f1')

    plt.title("Section 4.4 — Mean and Standard Deviation per Dataset",
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("section4_pixel_stats.png",
                    dpi=150, bbox_inches='tight')
        print("Saved: section4_pixel_stats.png")
import numpy as np
from matplotlib import pyplot as plt


def plot_difficulty_comparison(datasets_info, save=False):
    """
    Section 6.1 + 6.2
    Shows one sample image from each dataset
    side by side for direct visual comparison
    Grayscale vs Color clearly visible
    """
    fig, axes = plt.subplots(1, len(datasets_info), figsize=(10, 3))
    fig.suptitle(
        "Section 6.1 & 6.2 — Image Clarity and Grayscale vs Color",
        fontsize=12, fontweight='bold'
    )

    for i, (dataset, ds_name, color_label) in enumerate(datasets_info):
        img, label = dataset[0]
        img_np     = img.numpy()

        if img_np.shape[0] == 1:
            axes[i].imshow(img_np.squeeze(), cmap='gray')
        else:
            img_show = np.transpose(img_np, (1, 2, 0))
            img_show = (img_show - img_show.min()) / \
                       (img_show.max() - img_show.min())
            axes[i].imshow(img_show)

        axes[i].set_title(
            f"{ds_name}\n{color_label}",
            fontsize=10, fontweight='bold'
        )
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("section6_difficulty_comparison.png",
                    dpi=150, bbox_inches='tight')
        print("Saved: section6_difficulty_comparison.png")


def plot_complexity_table(save=False):
    """
    Section 6.3
    Justification table for increasing complexity order
    """
    data = [
        ["MNIST",         "28×28", "1",  "Grayscale", "Digits 0-9",
         "Clean, high contrast",          "Easy",   "1st"],
        ["Fashion-MNIST", "28×28", "1",  "Grayscale", "Clothing items",
         "Similar size, complex shapes",  "Medium", "2nd"],
        ["CIFAR-10",      "32×32", "3",  "Color",     "Natural objects",
         "Real photos, cluttered background", "Hard", "3rd"],
    ]

    columns = [
        "Dataset", "Size", "Channels", "Type",
        "Content", "Challenge", "Difficulty", "Order"
    ]

    fig, ax = plt.subplots(figsize=(18, 2.5))
    ax.axis('off')

    table = ax.table(
        cellText  = data,
        colLabels = columns,
        loc       = 'center',
        cellLoc   = 'center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.2)

    # header style
    for col in range(len(columns)):
        table[0, col].set_facecolor('#2c3e50')
        table[0, col].set_text_props(color='white', fontweight='bold')

    # alternating row colors
    row_colors = ['#f2f2f2', '#ffffff']
    for row in range(1, len(data) + 1):
        for col in range(len(columns)):
            table[row, col].set_facecolor(row_colors[row % 2])
        table[row, 0].set_text_props(fontweight='bold')
        table[row, 0].set_facecolor('#d0e8f1')

    # difficulty column colors
    diff_colors = {'Easy': '#d4edda', 'Medium': '#fff3cd', 'Hard': '#f8d7da'}
    diff_col_idx = columns.index("Difficulty")
    for row in range(1, len(data) + 1):
        diff_val = data[row - 1][diff_col_idx]
        table[row, diff_col_idx].set_facecolor(diff_colors[diff_val])

    plt.title(
        "Section 6.3 — Justification of Increasing Complexity Order",
        fontsize=12, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("section6_complexity_table.png",
                    dpi=150, bbox_inches='tight')
        print("Saved: section6_complexity_table.png")


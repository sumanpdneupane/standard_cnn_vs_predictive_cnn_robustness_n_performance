from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


def get_class_counts(dataset):
    labels = [label for _, label in dataset]
    counts = Counter(labels)
    sorted_counts = dict(sorted(counts.items()))
    return sorted_counts

def plot_class_distribution(ax, counts, class_names, title, color):
    classes = list(counts.keys())
    values = list(counts.values())

    bars = ax.bar(
        [class_names[c] for c in classes],
        values,
        color=color,
        edgecolor='black',
        linewidth=0.5
    )

    # add count on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f'{val:,}',
            ha='center',
            va='bottom',
            fontsize=7
        )

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Class", fontsize=9)
    ax.set_ylabel("Number of Samples", fontsize=9)
    ax.set_ylim(0, max(values) * 1.15)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


def plot_all_class_distributions(datasets_info, save=False):
    fig, axes = plt.subplots(1, len(datasets_info), figsize=(18, 5))
    fig.suptitle("Class Distribution — All Datasets", fontsize=13, fontweight='bold')

    for i, (counts, class_names, title, color) in enumerate(datasets_info):
        plot_class_distribution(
            ax=axes[i],
            counts=counts,
            class_names=class_names,
            title=title,
            color=color
        )

    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("section3_class_distribution.png",
                    dpi=150, bbox_inches='tight')
        print("Saved: section3_class_distribution.png")

def balance_analysis_table(datasets_info, save=False):
    rows = []
    for name, counts in datasets_info:
        values     = list(counts.values())
        min_count  = min(values)
        max_count  = max(values)
        mean_count = np.mean(values)
        imbalance  = max_count - min_count
        ratio      = max_count / min_count
        status     = "Balanced" if ratio < 1.2 else "Imbalanced"

        rows.append([
            name,
            f"{min_count:,}",
            f"{max_count:,}",
            f"{mean_count:,.0f}",
            f"{imbalance:,}",
            f"{ratio:.2f}",
            status
        ])

    columns = [
        "Dataset",
        "Min / Class",
        "Max / Class",
        "Mean / Class",
        "Imbalance",
        "Ratio",
        "Status"
    ]

    fig, ax = plt.subplots(figsize=(13, 2))
    ax.axis('off')

    table = ax.table(
        cellText    = rows,
        colLabels   = columns,
        loc         = 'center',
        cellLoc     = 'center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # header style
    for col in range(len(columns)):
        table[0, col].set_facecolor('#000000')
        table[0, col].set_text_props(color='white', fontweight='bold')

    # row styles
    row_colors = ['#f2f2f2', '#ffffff']
    for row in range(1, len(rows) + 1):
        for col in range(len(columns)):
            table[row, col].set_facecolor(row_colors[row % 2])

    # status column color
    for row in range(1, len(rows) + 1):
        status = rows[row - 1][-1]
        color  = '#d4edda' if status == "Balanced" else '#f8d7da'
        table[row, len(columns) - 1].set_facecolor(color)

    plt.title("Is the Dataset Balanced or Imbalanced?",
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    if save:
        plt.savefig("section3_balance_table.png", dpi=150, bbox_inches='tight')
    plt.show()
    if save:
        print("Saved: section3_balance_table.png")
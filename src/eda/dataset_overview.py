from matplotlib import pyplot as plt


def get_info(name, train_ds, test_ds):
    image, _ = train_ds[0]
    channels, h, w = image.shape
    return {
        "Dataset": name,
        "Train Samples": f"{len(train_ds):,}",
        "Test Samples": f"{len(test_ds):,}",
        "Classes": len(train_ds.classes),
        "Image Size": f"{h} x {w}",
        "Channels": channels,
        "Type": "Grayscale" if channels == 1 else "RGB",
        "Class Names": str(train_ds.classes)
    }

def plot_dataset_summary_table(rows, save=False):
    # remove Class Names column
    columns = ["Dataset", "Train Samples", "Test Samples",
               "Classes", "Image Size", "Channels", "Type"]

    data = [[row[col] for col in columns] for row in rows]

    fig, ax = plt.subplots(figsize=(13, 2))
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

    # header style
    for col in range(len(columns)):
        table[0, col].set_facecolor('#000000')
        table[0, col].set_text_props(color='white', fontweight='bold')

    # alternating row colors
    row_colors = ['#f2f2f2', '#ffffff']
    for row in range(1, len(data) + 1):
        for col in range(len(columns)):
            table[row, col].set_facecolor(row_colors[row % 2])

    # Type column color
    # Grayscale → light blue, Color → light orange
    type_col_idx = columns.index("Type")
    for row in range(1, len(data) + 1):
        type_val = data[row - 1][type_col_idx]
        color = '#d0e8f1' if type_val == "Grayscale" else '#fde8c8'
        table[row, type_col_idx].set_facecolor(color)

    plt.title("Section 1 — Dataset Summary",
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    if save:
        plt.savefig("section1_dataset_summary.png", dpi=150, bbox_inches='tight')
    plt.show()
    if save:
        print("Saved: section1_dataset_summary.png")

def plot_class_names_table(rows, save=False):
    # build data: Dataset + one column per class (0-9)
    col_headers = ["Dataset"] + [f"Class {i}" for i in range(10)]

    data = []
    for row in rows:
        class_names = eval(row["Class Names"]) if isinstance(row["Class Names"], str) else row["Class Names"]
        data.append([row["Dataset"]] + list(class_names))

    fig, ax = plt.subplots(figsize=(18, 2))
    ax.axis('off')

    table = ax.table(
        cellText  = data,
        colLabels = col_headers,
        loc       = 'center',
        cellLoc   = 'center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 2.0)

    # header style
    for col in range(len(col_headers)):
        table[0, col].set_facecolor('#000000')
        table[0, col].set_text_props(color='white', fontweight='bold')

    # alternating row colors
    row_colors = ['#f2f2f2', '#ffffff']
    for row in range(1, len(data) + 1):
        for col in range(len(col_headers)):
            table[row, col].set_facecolor(row_colors[row % 2])

    # dataset name column bold
    for row in range(1, len(data) + 1):
        table[row, 0].set_text_props(fontweight='bold')
        table[row, 0].set_facecolor('#d0e8f1')

    plt.title("Section 1 — Class Names per Dataset",
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    if save:
        plt.savefig("section1_class_names.png", dpi=150, bbox_inches='tight')
    plt.show()
    if save:
        print("Saved: section1_class_names.png")
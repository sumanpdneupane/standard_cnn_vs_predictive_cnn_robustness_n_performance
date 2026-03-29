import numpy as np
from matplotlib import pyplot as plt


def show_sample_images(dataset, title, class_names, save=False):
    # collect one image per class
    class_images = {}
    for image, label in dataset:
        if label not in class_images:
            class_images[label] = image
        if len(class_images) == len(class_names):
            break

    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 2.5))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    for i, class_idx in enumerate(sorted(class_images.keys())):
        img = class_images[class_idx]

        # convert tensor to numpy for display
        # MNIST / Fashion-MNIST: (1, H, W) → (H, W)
        # CIFAR-10:              (3, H, W) → (H, W, 3)
        img_np = img.numpy()
        if img_np.shape[0] == 1:
            # grayscale
            axes[i].imshow(img_np.squeeze(), cmap='gray')
        else:
            # color: move channel from first to last dim
            axes[i].imshow(np.transpose(img_np, (1, 2, 0)))

        axes[i].set_title(class_names[class_idx], fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(f"section2_{title.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()
    if save:
        print("Saved: section2_all_datasets.png")


def combine_show_sample_images(datasets_info, save=False):
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    fig.suptitle("Sample Images — All Datasets", fontsize=13,
                 fontweight='bold')

    for row, (dataset, class_names, ds_name) in enumerate(datasets_info):

        # collect one image per class
        class_images = {}
        for image, label in dataset:
            if label not in class_images:
                class_images[label] = image
            if len(class_images) == 10:
                break

        for col, class_idx in enumerate(sorted(class_images.keys())):
            img = class_images[class_idx]
            img_np = img.numpy()

            if img_np.shape[0] == 1:
                axes[row, col].imshow(img_np.squeeze(), cmap='gray')
            else:
                axes[row, col].imshow(np.transpose(img_np, (1, 2, 0)))

            # top row: show class name
            if row == 0:
                axes[row, col].set_title(
                    class_names[class_idx], fontsize=7
                )

            axes[row, col].axis('off')

            # show dataset name on left side AFTER axis('off')
            # using fig.text for reliable left side placement
            if col == 0:
                axes[row, col].text(
                    x=-0.3,  # left of the image
                    y=0.5,  # vertically centered
                    s=ds_name,  # dataset name
                    fontsize=10,
                    fontweight='bold',
                    ha='right',  # align text to the right
                    va='center',  # align text to center
                    transform=axes[row, col].transAxes
                )

    plt.tight_layout()
    if save:
        plt.savefig("section2_all_datasets.png", dpi=150, bbox_inches='tight')
    plt.show()
    if save:
        print("Saved: section2_all_datasets.png")

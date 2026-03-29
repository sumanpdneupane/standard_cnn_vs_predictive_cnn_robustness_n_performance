import numpy as np
from matplotlib import pyplot as plt

def show_grid(images, rows, cols, title=""):
    fig = plt.figure(figsize=(cols * 2, rows * 2))
    for i, img in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        imshow(img)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
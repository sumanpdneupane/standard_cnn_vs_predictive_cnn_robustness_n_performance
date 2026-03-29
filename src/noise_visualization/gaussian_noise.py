import torch

sigmas = [0.0, 0.1, 0.2, 0.3, 0.5]

def _add_gaussian_noise(image, sigma=0.1):
    noise = torch.randn_like(image) * sigma
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0., 1.)
    return noisy_image

def add_noise_to_dataset(imgs, sigmas):
    img_noisy = []
    for sigma in sigmas:
        for img in imgs:
            img_noisy.append(_add_gaussian_noise(img, sigma))
    return img_noisy

def get_samples(dataset, n=5):
    imgs = []
    for i in range(n):
        img, _ = dataset[i]
        # Convert grayscale to 3-channel for consistent display
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        imgs.append(img)
    return imgs


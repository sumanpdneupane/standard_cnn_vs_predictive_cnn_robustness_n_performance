import torch

ps = [0.0, 0.05, 0.1, 0.2, 0.3]

def _add_salt_pepper_noise(image, p=0.1):
    noisy = image.clone()
    rand = torch.rand_like(image)

    # Pepper (0)
    noisy[rand < (p / 2)] = 0.0

    # Salt (1)
    noisy[rand > 1 - (p / 2)] = 1.0

    return noisy

def add_noise_to_dataset(imgs, ps):
    img_noisy = []
    for p in ps:
        for img in imgs:
            img_noisy.append(_add_salt_pepper_noise(img, p))
    return img_noisy

def get_samples(dataset, n=5):
    imgs = []
    for i in range(n):
        img, _ = dataset[i]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        imgs.append(img)
    return imgs

import numpy as np

def salt_pepper_noise(img, salt_probability, pepper_probability):
    img_copy = np.copy(img)
    salt = np.random.rand(*img.shape) < salt_probability
    pepper = np.random.rand(*img.shape) < pepper_probability
    img_copy[salt] = 1
    img_copy[pepper] = 0
    return img_copy

def random_noise(img, factor):
    img_copy = np.copy(img)
    img_copy = img_copy + (np.random.rand(*img_copy.shape)/10)*factor
    return img_copy

def gaussian_noise(img, mean, std):
    img_copy = np.copy(img)
    img_copy = img_copy + np.random.normal(mean, std, *img_copy.shape)
    return img_copy
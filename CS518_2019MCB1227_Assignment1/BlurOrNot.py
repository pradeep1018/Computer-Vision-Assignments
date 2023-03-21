from skimage.io import imread
import numpy as np
from skimage.color import rgb2gray
from scipy.stats import poisson

def convolve(image, filter):
    image_convolve = np.zeros((image.shape[0]-2,image.shape[1]-2))
    for i in range(image_convolve.shape[0]):
        for j in range(image_convolve.shape[1]):
            image_convolve[i][j] = (filter*image[i:i+3, j:j+3]).sum()
    return image_convolve

def laplacian(image):
    filter = np.array([[0., 1., 0.],[1., -4., 1.],[0., 1., 0.]])
    laplacian_image = convolve(image, filter)
    return laplacian_image

def BlurorNot(image_path):
    image = imread(image_path)
    image = rgb2gray(image)*255.
    laplacian_image = laplacian(image)
    val = np.var(laplacian_image)
    if val >= 100:
        blurornot = 0
    else:
        blurornot = 1
    prob_blur = 1-poisson.cdf(k=val, mu=100)
    return blurornot, prob_blur
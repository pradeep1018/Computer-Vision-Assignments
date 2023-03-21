import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import numpy as np

def gaussian_filter(image):
    x, y = np.mgrid[-1:2,-1:2]
    normal = 1/(2.0*np.pi)
    filter =  np.exp(-((x**2+y**2)/2.0))*normal
    return filter

def convolve(image, filter):
    image_convolve = np.zeros((image.shape[0]-2,image.shape[1]-2))
    for i in range(image_convolve.shape[0]):
        for j in range(image_convolve.shape[1]):
            image_convolve[i][j] = (filter*image[i:i+3, j:j+3]).sum()
    return image_convolve

def gaussian(image):
    filter = gaussian_filter(image)
    smooth_image = convolve(image, filter)
    return smooth_image

def sobel(image):
    Fx = np.array([[-1., 0., 1.],[-2., 0., 2.],[-1., 0., 1.]])
    Fy = np.array([[1., 2., 1.],[0., 0., 0.],[-1., -2., -1.]])
    Gx = convolve(image, Fx)
    Gy = convolve(image, Fy)
    G = np.hypot(Gx, Gy)
    G = (G/G.max())*255
    theta = np.arctan2(Gy, Gx)
    return G, theta

def non_maximum_suppression(image, angle):
    image_suppress = np.zeros((image.shape[0],image.shape[1]))
    angle = angle*180./np.pi
    angle[angle<0] += 180
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            if 0 <= angle[i,j] < 22.5 or 157.5 <= angle[i,j] <= 180:
                n1 = image[i,j+1]
                n2 = image[i,j-1]
            elif 22.5 <= angle[i,j] < 67.5:
                n1 = image[i+1,j-1]
                n2 = image[i-1,j+1]
            elif 67.5 <= angle[i,j] < 112.5:
                n1 = image[i+1,j]
                n2 = image[i-1,j]
            else:
                n1 = image[i-1,j-1]
                n2 = image[i+1,j+1]
            if image[i,j] >= n1 and image[i,j] >= n2:
                image_suppress[i,j] = image[i,j]
    return image_suppress

def hysteresis_thresholding(image, Low_Threshold, High_Threshold):
    weak = 127
    strong = 255
    high = image.max()*High_Threshold
    low = image.max()*Low_Threshold
    image_hysteresis = np.zeros((image.shape[0],image.shape[1]))
    strong_x, strong_y = np.where(image >= high)
    weak_x, weak_y = np.where((image < high) & (image >= low))
    image_hysteresis[strong_x, strong_y] = strong
    image_hysteresis[weak_x, weak_y] = weak
    for i in range(1, image_hysteresis.shape[0]-1):
        for j in range(1, image_hysteresis.shape[1]-1):
            if (image_hysteresis[i,j] == weak):
                if ((image_hysteresis[i+1, j-1] == strong) or (image_hysteresis[i+1, j] == strong) or 
                (image_hysteresis[i+1, j+1] == strong) or (image_hysteresis[i, j-1] == strong) or
                (image_hysteresis[i, j+1] == strong) or (image_hysteresis[i-1, j-1] == strong) or 
                (image_hysteresis[i-1, j] == strong) or (image_hysteresis[i-1, j+1] == strong)):
                    image_hysteresis[i, j] = strong
                else:
                    image_hysteresis[i, j] = 0
    return image_hysteresis
    
def image_postprocess(image):
    final_image = np.zeros((image.shape[0]+4,image.shape[1]+4))
    for i in range(final_image.shape[0]):
        for j in range(final_image.shape[1]):
            if (i <= 1 or i >= final_image.shape[0] - 2) or (j <= 1 or j >= final_image.shape[1] - 2):
                final_image[i][j] = 0
            else:
                final_image[i][j] = image[i-2][j-2]
    return final_image

def myCannyEdgeDetector(image, Low_Threshold=0.05, High_Threshold=0.08):
    smooth_image = gaussian(image) 
    edge_gradient, direction = sobel(smooth_image)
    image_suppress = non_maximum_suppression(edge_gradient, direction)
    canny_image = hysteresis_thresholding(image_suppress, Low_Threshold, High_Threshold)
    canny_image = image_postprocess(canny_image)
    return canny_image

image_paths = ['/image3.jpg', '/image4.jpg', '/image5.jpg']
for i, image_path in enumerate(image_paths):
    image = imread('TestImages'+image_path)
    image = rgb2gray(image)

    skimage_canny_image = canny(image)
    my_canny_image = myCannyEdgeDetector(image)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(skimage_canny_image, cmap='gray')
    ax[0].set_title('Canny Edge Detection using skimage function')
    ax[1].imshow(my_canny_image, cmap='gray')
    ax[1].set_title('Canny Edge Detection using myCannyEdgeDetector function')
    plt.show()

    print('Image',i+1)
    print('The peak signal to noise ratio (PSNR) value is ', peak_signal_noise_ratio(skimage_canny_image, my_canny_image.astype(bool)))
    print('The Structural Similarity Index Metric (SSIM) value is ', structural_similarity(skimage_canny_image, my_canny_image.astype(bool)))
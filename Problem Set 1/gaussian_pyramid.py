import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def cross_correlation_2d(X,K):
    height,width = K.shape
    Y = np.zeros((X.shape[0]-height+1,X.shape[1]-width+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j]=(X[i:i+height, j:j+width]*K).sum()
    return Y

def convolve_2d(X,K):
    height,width = K.shape
    X = np.flip(np.flip(X, axis=0), axis=1)
    Y = np.zeros((X.shape[0]-height+1,X.shape[1]-width+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j]=(X[i:i+height, j:j+height]*K).sum()
    return np.flip(np.flip(Y, axis=0), axis=1)

def gaussian_blur_kernel_2d(kernel_size=3,sigma=1):
    kernel=np.zeros((kernel_size,kernel_size))
    r = kernel_size//2
    for x in range(-r,r+1):
        for y in range(-r,r+1):
            kernel[x+r,y+r] = 1/(2*np.pi*sigma**2)*np.exp(-1/(2*sigma**2)*(x**2+y**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def low_pass(image, kernel_size=3,sigma=1):
    kernel = gaussian_blur_kernel_2d(kernel_size,sigma)
    return convolve_2d(image,kernel)

def image_subsampling(image):
    return image[::2,::2]

def gaussian_pyramid(image, kernel_size=3,sigma=1):
    for i in range(1,4):
        image_array = np.array(image)
        shape=image_array.shape[:2]
        red_channel = image_array[:, :, 0].reshape(shape)
        green_channel = image_array[:, :, 1].reshape(shape)
        blue_channel = image_array[:, :, 2].reshape(shape)
        red_channel_blurred = image_subsampling(low_pass(red_channel,kernel_size,sigma))
        green_channel_blurred = image_subsampling(low_pass(green_channel,kernel_size,sigma))
        blue_channel_blurred = image_subsampling(low_pass(blue_channel,kernel_size,sigma))
        combined_image_array = np.stack((red_channel_blurred, green_channel_blurred, blue_channel_blurred), axis=-1)
        image = Image.fromarray(combined_image_array.astype(np.uint8))
        image.save("./resolution"+str(i)+".png","PNG")


image = Image.open("./Vangogh.png")

if __name__=="__main__":
    gaussian_pyramid(image,5,1000)
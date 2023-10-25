# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import scipy, cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_gray(img):
    if len(img.shape) == 3:
        gray_image = np.mean(img, axis=2).astype(np.float64)
    else:
        gray_image = img
    return gray_image
def gaussian_blur_kernel_2d(kernel_size=3,sigma=1):
    kernel=np.zeros((kernel_size,kernel_size))
    r = kernel_size//2
    for x in range(-r,r+1):
        for y in range(-r,r+1):
            kernel[x+r,y+r] = 1/(2*np.pi*sigma**2)*np.exp(-1/(2*sigma**2)*(x**2+y**2))
    kernel = kernel / np.sum(kernel)
    return kernel
def convolve_2d(X,K):
    height,width = K.shape
    X = np.flip(np.flip(X, axis=0), axis=1)
    Y = np.zeros((X.shape[0]-height+1,X.shape[1]-width+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j]=(X[i:i+height, j:j+height]*K).sum()
    return np.flip(np.flip(Y, axis=0), axis=1)
def gradient_x(img):
    # convert img to grayscale
    # should we use int type to calclate gradient?
    # should we conduct some pre-processing to remove noise? which kernel should we pply?
    # which kernel should we choose to calculate gradient_x?
    # TODO
    gray_image = rgb_to_gray(img)
    filtered_image = scipy.ndimage.gaussian_filter(gray_image, sigma = 1, mode = 'reflect')
    return scipy.ndimage.sobel(filtered_image, axis = 0, mode = 'reflect')

def gradient_y(img):
    # TODO
    gray_image = rgb_to_gray(img)
    filtered_image = scipy.ndimage.gaussian_filter(gray_image, sigma = 1, mode = 'reflect')
    return scipy.ndimage.sobel(filtered_image, axis = 1, mode = 'reflect')

def harris_response(img, alpha=0.05, win_size=3):
    # In this function you are going to claculate harris response R.
    # Please refer to 04_Feature_Detection.pdf page 29 for details. 
    # You have to discover how to calculate det(M) and trace(M), and
    # remember to smooth the gradients. 
    # Avoid using too much "for" loops to speed up.
    # TODO
    win_size = win_size // 2
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    sigma = win_size / 7
    kernel = gaussian_blur_kernel_2d(win_size, sigma)
    A = grad_x * grad_x
    B = grad_x * grad_y
    C = grad_y * grad_y
    R = np.zeros_like(A)
    for i in range(win_size, img.shape[0] - win_size):
        for j in range(win_size, img.shape[1] - win_size):
            windows_A = np.sum(A[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1] * kernel)
            windows_B = np.sum(B[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1] * kernel)
            windows_C = np.sum(C[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1] * kernel)
            det_R = windows_A * windows_C - windows_B * windows_B
            tr_R = windows_A + windows_C
            windows_R = det_R - sigma * tr_R**2
            R[i, j] = windows_R
    return R

def corner_selection(R, th=0.01, min_dist=8):
    # non-maximal suppression for R to get R_selection and transform selected corners to list of tuples
    # hint: 
    #   use ndimage.maximum_filter()  to achieve non-maximum suppression
    #   set those which aren’t **local maximum** to zero.
    # TODO
    width, height = R.shape
    threshold = th * np.max(R)
    condition_matrix = (R > threshold).astype(np.uint8)
    pix = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if condition_matrix[i, j]:
                window = R[np.clip(i-min_dist//2,0,width-1):np.clip(i+min_dist//2,0,width-1), np.clip(j-min_dist//2,0,height-1):np.clip(j+min_dist//2,0,height-1)]
                if R[i,j]-1e-6<np.max(window) < R[i,j] + 1e-6:
                    pix.append((i,j))
    return list(pix)

def histogram_of_gradients(img, pix):
    # no template for coding, please implement by yourself.
    # You can refer to implementations on Github or other websites
    # Hint: 
    #   1. grad_x & grad_y
    #   2. grad_dir by arctan function
    #   3. for each interest point, choose m*m blocks with each consists of m*m pixels
    #   4. I divide the region into n directions (maybe 8).
    #   5. For each blocks, calculate the number of derivatives in those directions and normalize the Histogram. 
    #   6. After that, select the prominent gradient and take it as principle orientation.
    #   7. Then rotate it’s neighbor to fit principle orientation and calculate the histogram again. 
    # TODO
    WIDTH,HEIGHT = img.shape
    block_size = 4  # 块大小
    num_bins = 8  # 方向直方图的箱数
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    grad_magnitude = np.sqrt(grad_x**2,grad_y**2)
    grad_direction = np.arctan2(grad_y,grad_x)*180.0/np.pi + 180.0
    features = []
    for cornel in pix:
        cornel_x, cornel_y = cornel
        # 选角点周围的区域
        # 先算出主方向
        temp_feature = [0 for i in range(num_bins)]
        for i in range(cornel_x - block_size**2//2 + 1, cornel_x + block_size**2//2 + 1):
            for j in range(cornel_y - block_size**2//2 + 1, cornel_y + block_size**2//2 + 1):
                    temp_feature[int(grad_direction[np.clip(i,0,WIDTH-1),np.clip(j,0,HEIGHT-1)]*num_bins/360)]+=grad_magnitude[np.clip(i,0,WIDTH-1),int(np.clip(j,0,HEIGHT-1))]
        orientation = np.argmax(temp_feature)*np.pi/4
        change = np.array([[np.cos(orientation),-np.sin(orientation)],[np.sin(orientation),np.cos(orientation)]])
        feature = []
        #print(change)
        # 从右上角往下遍历, 旋转
        for index_x in range(-block_size//2,block_size//2):
            for index_y in range(-block_size//2,block_size//2):
                temp_feature = [0 for i in range(num_bins)]
                for i in range(cornel_x + index_x * block_size + 1, cornel_x + (index_x + 1) * block_size + 1):
                    for j in range(cornel_y + index_y * block_size + 1, cornel_y + (index_y + 1) * block_size + 1):
                        new_point = np.array(change@[i,j],dtype=np.uint8)
                        x,y=new_point
                        temp_feature[int(grad_direction[np.clip(x,0,WIDTH-1),np.clip(y,0,HEIGHT-1)]*num_bins/360)]+=grad_magnitude[np.clip(x,0,WIDTH-1),np.clip(y,0,HEIGHT-1)]
                feature.append(temp_feature)
                #print(temp_feature)
        # 归一化
        feature = feature / np.sum(feature)
        features.append(feature)
    return features


        
    

def feature_matching(img_1, img_2):
    # align two images using \verb|harris_response|, \verb|corner_selection|,
    # \verb|histogram_of_gradients|
    # hint: calculate distance by scipy.spacial.distance.cdist (using HoG features, euclidean will work well)
    # TODO
    THRESHOLD =0.5
    corners_1 = corner_selection(harris_response(img_1,0.05,3),0.01,3)
    corners_2 = corner_selection(harris_response(img_2,0.05,3),0.01,3)
    print("start hog")
    features_1 = histogram_of_gradients(img_1, corners_1)
    features_2 = histogram_of_gradients(img_2, corners_2)

    pix_1 = []
    pix_2 = []
    for i, feature_1 in enumerate(features_1):
        for j, feature_2 in enumerate(features_2):
            bias = np.sqrt(np.sum((feature_1-feature_2)**2))
            if bias < THRESHOLD:
                pix_1.append(corners_1[i])
                pix_2.append(corners_2[j])
    return pix_1, pix_2

def compute_homography(pixels_1, pixels_2):
    # compute the best-fit homography using the Singular Value Decomposition (SVD)
    # homography matrix is a (3,3) matrix consisting rotation, translation and projection information.
    # consider how to form matrix A for U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    # homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    # TODO
    return homo_matrix

def align_pair(pixels_1, pixels_2):
    # utilize \verb|homo_coordinates| for homogeneous pixels
    # and \verb|compute_homography| to calulate homo_matrix
    # implement RANSAC to compute the optimal alignment.
    # you can refer to implementations online.
    return est_homo

def stitch_blend(img_1, img_2, est_homo):
    # hint: 
    # First, project four corner pixels with estimated homo-matrix
    # and converting them back to Cartesian coordinates after normalization.
    # Together with four corner pixels of the other image, we can get the size of new image plane.
    # Then, remap both image to new image plane and blend two images using Alpha Blending.
    # TODO
    return est_img


def generate_panorama(ordered_img_seq):
    # finally we can use \verb|feature_matching| \verb|align_pair| and \verb|stitch_blend| to generate 
    # panorama with multiple images
    # TODO
    return est_panorama

if __name__ == '__main__':
    # make image list
    # call generate panorama and it should work well
    # save the generated image following the requirements
    file_path = './Problem_Set_2/Problem2Images/1_1.jpg'
    # image = cv2.imread(file_path)
    # image = rgb_to_gray(image)
    # R = harris_response(image)
    # print(histogram_of_gradients(image,corner_selection(harris_response(image))))
    # exit()
    # file_path_2 = './Problem_Set_2/Problem2Images/1_2.jpg'
    image = cv2.imread(file_path)
    # image_2 = cv2.imread(file_path_2)
    # image = rgb_to_gray(image)
    # image_2 = rgb_to_gray(image_2)
    # pix_1,pix_2 = feature_matching(image,image_2)
    # print(len(pix_1))
    # for i in pix_1:
    #     image[i[0],i[1]]=255
    # for i in pix_2:
    #     image_2[i[0],i[1]]=255
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(image, cmap='gray')
    # axes[0].set_title('Image 1')

    # # 在第二个子图上显示第二幅图像
    # axes[1].imshow(image_2, cmap='gray')
    # axes[1].set_title('Image 2')
    # for i in range(len(pix_1)):
    #     plt.plot([pix_1[i],pix_2[i]],'ro-')
    # plt.show()
    # grad_x = gradient_x (image)
    # grad_y = gradient_y (image)
    # out = np.sqrt(grad_x**2+grad_y**2)
    # plt.imshow(out,cmap='gray')
    # plt.show()
    R = harris_response(image,0.05,3)
    target = corner_selection(R,0.01,3)
    output = np.zeros_like(R)
    image = rgb_to_gray(image)
    for i in target:
        # image[i[0],i[1]]=255
        plt.plot(i[0],i[1],color='red')
    plt.imshow(image)
    plt.show()
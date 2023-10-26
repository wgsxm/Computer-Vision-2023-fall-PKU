# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import scipy, cv2
import numpy as np
import matplotlib.pyplot as plt
from random import sample
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
    filtered_image = scipy.ndimage.gaussian_filter(img, sigma = 1, mode = 'reflect')
    return scipy.ndimage.sobel(filtered_image, axis = 0, mode = 'reflect')

def gradient_y(img):
    # TODO
    filtered_image = scipy.ndimage.gaussian_filter(img, sigma = 1, mode = 'reflect')
    return scipy.ndimage.sobel(filtered_image, axis = 1, mode = 'reflect')

def harris_response(img, alpha=0.05, win_size=3):
    # In this function you are going to claculate harris response R.
    # Please refer to 04_Feature_Detection.pdf page 29 for details. 
    # You have to discover how to calculate det(M) and trace(M), and
    # remember to smooth the gradients. 
    # Avoid using too much "for" loops to speed up.
    # TODO
    kernel = gaussian_blur_kernel_2d(win_size, win_size/7)
    A = grad_x * grad_x
    win_size = win_size // 2
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
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
            windows_R = det_R - alpha * tr_R**2
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
        orientation = np.argmax(temp_feature)*360/num_bins
        feature = []
        #print(change)
        # 从右上角往下遍历, 旋转
        for index_x in range(-block_size//2,block_size//2):
            for index_y in range(-block_size//2,block_size//2):
                temp_feature = [0 for i in range(num_bins)]
                for i in range(cornel_x + index_x * block_size + 1, cornel_x + (index_x + 1) * block_size + 1):
                    for j in range(cornel_y + index_y * block_size + 1, cornel_y + (index_y + 1) * block_size + 1):
                        temp_angle = grad_direction[np.clip(i,0,WIDTH-1),np.clip(j,0,HEIGHT-1)] - orientation
                        if temp_angle<0:
                             temp_angle+=360
                        temp_feature[int(temp_angle*num_bins/360)]+=grad_magnitude[np.clip(i,0,WIDTH-1),np.clip(j,0,HEIGHT-1)]
                feature+=temp_feature
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
    THRESHOLD =0.15
    corners_1 = corner_selection(harris_response(img_1,0.05,3),0.01,3)
    corners_2 = corner_selection(harris_response(img_2,0.05,3),0.01,3)
    features_1 = histogram_of_gradients(img_1, corners_1)
    features_2 = histogram_of_gradients(img_2, corners_2)

    pix_1 = []
    pix_2 = []
    for i, feature_1 in enumerate(features_1):
        temp_point = []
        temp_bias = []
        for j, feature_2 in enumerate(features_2):
            bias = np.sqrt(np.sum((feature_1-feature_2)**2))
            if bias < THRESHOLD:
                temp_point.append(corners_2[j])
                temp_bias.append(bias)
        if len(temp_point):
            pix_1.append(corners_1[i])
            pix_2.append(temp_point[np.argmin(temp_bias)])
    return pix_1, pix_2

def compute_homography(pixels_1, pixels_2):
    # compute the best-fit homography using the Singular Value Decomposition (SVD)
    # homography matrix is a (3,3) matrix consisting rotation, translation and projection information.
    # consider how to form matrix A for U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    # homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    # TODO
    A = []
    for p1, p2 in zip(pixels_1, pixels_2):
        x1, y1 = p1
        x2, y2 = p2
        A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2, -x2])
        A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2, -y2])
    A = np.array(A)
    # 使用SVD分解
    U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    return homo_matrix

def align_pair(pixels_1, pixels_2):
    
    # utilize \verb|homo_coordinates| for homogeneous pixels
    # and \verb|compute_homography| to calulate homo_matrix
    # implement RANSAC to compute the optimal alignment.
    # you can refer to implementations online.
    num_iterations = 200  # Number of RANSAC iterations
    inlier_threshold = 5.0  # Threshold for considering a point as an inlier
    num_picked = 4
    best_inliers = []  # Best set of inliers
    best_est_homo = None  # Best estimated homography matrix

    for _ in range(num_iterations):
        # Randomly sample 4 point correspondences
        sample_indices = sample(range(len(pixels_1)), num_picked)
        sampled_pixels_1 = [pixels_1[i] for i in sample_indices]
        sampled_pixels_2 = [pixels_2[i] for i in sample_indices]

        # Calculate homography matrix using the sampled points
        est_homo = compute_homography(sampled_pixels_1, sampled_pixels_2)

        # Transform all pixels from image 1 using the estimated homography
        homo_coordinates_1 = np.array(pixels_1 + [(1,) for _ in range(len(pixels_1))])
        transformed_pixels = np.dot(est_homo, homo_coordinates_1.T).T
        transformed_pixels = transformed_pixels[:, :2] / transformed_pixels[:, 2, None]

        # Calculate the Euclidean distance between the transformed pixels and image 2 pixels
        distances = np.linalg.norm(transformed_pixels - pixels_2, axis=1)

        # Count the number of inliers based on the threshold
        inliers = np.where(distances < inlier_threshold)[0]

        # Update the best set of inliers and homography matrix if this is the best so far
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_est_homo = est_homo

    return best_est_homo

def stitch_blend(img_1, img_2, est_homo):
    # hint: 
    # First, project four corner pixels with estimated homo-matrix
    # and converting them back to Cartesian coordinates after normalization.
    # Together with four corner pixels of the other image, we can get the size of new image plane.
    # Then, remap both image to new image plane and blend two images using Alpha Blending.
    # TODO
    W, H= img_1.shape[:2]
    corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]])
    projected_corners = np.dot(est_homo, corners.T).T
    projected_corners /= projected_corners[:, 2, np.newaxis]
    min_x = np.min(projected_corners[:,1])
    max_x = np.max(projected_corners[:,1])
    min_y = np.min(projected_corners[:,0])
    max_y = np.max(projected_corners[:,0])
    width,height = img_2.shape
    alpha = 0.45
    if min_x < 0:
        height -= min_x
    if min_y < 0:
        width +=min_y   
    print(img_1.shape)
    print(img_2.shape)
    print(min_x,min_y)
    width = max(width,max_y)
    height = max(height,max_x)
    print(width,height)
    est_img = np.zeros((int(width)+1,int(height)+1))
    for i in range(W):
        for j in range(H):
            x,y = i,j
            py,px=0,0
            if min_x>0:
                y = int(y+min_x)
            else:
                px = -min_x
            if min_y>0:
                x = int(x+min_y)
            else:
                py = -min_y
            if y>px and py <= x <= img_2.shape[0]+py:
                 est_img[x,y]=img_1[i,j]*alpha
            else:
                est_img[x,y]=\
                img_1[i,j]
    for i in range(img_2.shape[0]-1):
        for j in range(img_2.shape[1]-1):
            x,y=i,j
            px,py=0,0
            if min_x<0:
                y-=min_x
            else:
                px =min_x
            if min_y<0:
                x-=min_y
            else:
                py =min_y
            x = int(x)
            y = int(y)
            if y<px+img_1.shape[1] and py <= x <= img_2.shape[0]+py:
                 est_img[x,y]+=img_2[i,j]*(1-alpha)
            else:
                est_img[x,y]+=img_2[i,j]
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
    pass
import cv2
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import trimesh

def normalize_disparity_map(disparity_map):
    # Normalize disparity map for visualization
    # disparity should be larger than zero
    return np.maximum(disparity_map, 0.0) / disparity_map.max()
    
def visualize_disparity_map(disparity_map, gt_map):
    # Normalize disparity map
    disparity_map = normalize_disparity_map(disparity_map)
    gt_map = normalize_disparity_map(gt_map)
    # Visualize
    concat_map = np.concatenate([disparity_map, gt_map], axis=1)
    plt.imshow(concat_map, 'gray')
    plt.show()

def compute_disparity_map_simple(ref_image, sec_image, window_size, disparity_range, matching_function):
    # 1. Simple Stereo System: image planes are parallel to each other
    # 2. For each row, scan all pixels in that row
    # 3. Generate a window for each pixel
    # 4. Search a disparity(d) in (min_disparity, max_disparity)
    # 5. Select the best disparity that minimize window difference between (row, col) and (row, col - d)

    return disparity_map

def simple_disparity(ref_image, second_image, gt_map):
    # 1. Change window size, disparity range and matching functions
    # 2. Report the disparity maps and running time
    
    window_sizes = []  # Try different window sizes
    disparity_range = (, )  # Determine appropriate disparity range
    matching_functions = ['SSD', 'SAD', 'normalized_correlation']  # Try different matching functions
    
    # Generate disparity maps for different settings
    for window_size in window_sizes:
        for matching_function in matching_functions:
            disparity_map = compute_disparity_map_simple(ref_image, second_image, window_size, disparity_range, matching_function)
            visualize_disparity_map(disparity_map, gt_map)

def compute_depth_map(disparity_map, baseline, focal_length):
    # 1. Compute depth map by z = fB / (x - x')


    return depth_map

def visualize_pointcloud(ref_image, disparity_map):
    # 1. Calculate depth map from disparity
    # 2. Set pointcloud's XYZ as image's XY and depth
    # 3. Set pointcloud's color as ref_image
    # 4. Visualize pointcloud by trimesh, We recommend using MeshLab to open ply file
    # 6. Adjust the baseline and focal_length for better performance
    # 7. You may need to cut some outliers
    
    baseline = 10
    focal_length = 10
    depth_map = compute_depth_map(disparity_map, baseline, focal_length)

    # Points

    # Colors

    # Save pointcloud as ply file
    pointcloud = trimesh.PointCloud(points, colors)
    pointcloud.export("pointcloud.ply", file_type="ply")
    
def compute_disparity_map_dp(ref_image, sec_image):
    # Dynamic programming stereo matching

    return disparity_map

if __name__ == "__main__":
    # Read images
    moebius_image1 = cv2.imread("data/moebius1.png")
    moebius_image1_gray = cv2.cvtColor(moebius_image1, cv2.COLOR_BGR2GRAY)
    moebius_image2 = cv2.imread("data/moebius2.png")
    moebius_image2_gray = cv2.cvtColor(moebius_image2, cv2.COLOR_BGR2GRAY)
    moebius_gt = cv2.imread("data/moebius_gt.png", cv2.IMREAD_GRAYSCALE)

    tsukuba_image1 = cv2.imread("data/tsukuba1.jpg")
    tsukuba_image1_gray = cv2.cvtColor(tsukuba_image1, cv2.COLOR_BGR2GRAY)
    tsukuba_image2 = cv2.imread("data/tsukuba2.jpg")
    tsukuba_image2_gray = cv2.cvtColor(tsukuba_image2, cv2.COLOR_BGR2GRAY)
    tsukuba_gt = cv2.imread("data/tsukuba_gt.jpg", cv2.IMREAD_GRAYSCALE)


    # Task 0: Visualize cv2 Results
    stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
    moebius_disparity_cv2 = stereo.compute(moebius_image1_gray, moebius_image2_gray)
    visualize_disparity_map(moebius_disparity_cv2, moebius_gt)
    tsukuba_disparity_cv2 = stereo.compute(tsukuba_image1_gray, tsukuba_image2_gray)
    visualize_disparity_map(tsukuba_disparity_cv2, tsukuba_gt)


    # Task 1: Simple Disparity Algorithm
    simple_disparity(tsukuba_image1_gray, tsukuba_image2_gray, tsukuba_gt)
    # This may run for a long time! 
    simple_disparity(moebius_image1_gray, moebius_image2_gray, moebius_gt)

    # Task 2: Compute depth map and visualize 3D pointcloud
    # You can use the gt/cv2 disparity map first, then try your own disparity map
    visualize_pointcloud(tsukuba_image1, tsukuba_disparity)
    visualize_pointcloud(moebius_image1, moebius_disparity)

    # Task 3: Non-local constraints
    compute_disparity_map_dp(tsukuba_image1_gray, tsukuba_image2_gray)



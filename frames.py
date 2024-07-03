"""
Utility functions to manipulate image frames.
"""
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse

def compute_left_disparity_map(img_left, img_right, verbose=False):
    """
    Compute the disparity map from a stereo image pair.

    Args:
        img_left (numpy.ndarray): the left stereo image.
        img_right (numpy.ndarray): the right stereo image.
        verbose (bool, optional): tag to toggle debug information. Defaults to False.

    Returns:
        numpy.ndarray: the disparity map.
    """

    # compute the disparity map
    matcher = cv2.StereoSGBM_create(numDisparities=96, minDisparity=0, blockSize=11,
                                    P1 = 8*3*6**2, P2 = 32*3*6**2,
                                    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    start = datetime.datetime.now()
    disparity = matcher.compute(img_left, img_right).astype(np.float32) / 16.0
    end = datetime.datetime.now()

    if verbose:
        print('Time taken to compute disparity:', end - start)

    return disparity

def decompose_projection_matrix(p):
    """
    Decompose the projection matrix into intrinsic and extrinsic parameters.

    Args:
        p (numpy.ndarray): the projection matrix.

    Returns:
        k (numpy.ndarray): the intrinsic matrix.
        r (numpy.ndarray): the rotation matrix.
        t (numpy.ndarray): the translation vector. 
    """
    
    # decompose the projection matrix
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]

    return k, r, t

def calc_depth_map(disp_left, k_left, t_left, t_right):
    """
    Calculate the depth map from the disparity map.

    Args:
        disp_left (numpy.ndarray): the disparity map.
        k_left (numpy.ndarray): the intrinsic matrix of the left camera.
        t_left (numpy.ndarray): the translation vector of the left camera.
        t_right (numpy.ndarray): the translation vector of the right camera.

    Returns:
        numpy.ndarray: the depth map.
    """

    # get the focal length of the x axis for the left camera
    f = k_left[0, 0]

    # calculate the baseline
    b = t_right[0] - t_left[0]

    # avoid instability and division by zero
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1

    # calculate the depth map
    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left

    return depth_map

def plot_depth_hist(depth_map):
    """
    Generate a histogram of the depth map.

    Args:
        depth_map (numpy.ndarray): the depth map.
    """

    plt.figure(figsize=(10, 10))
    plt.hist(depth_map.flatten())
    plt.title('Depth map histogram')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def stereo_2_depth(img_left, img_right, P0, P1):
    """
    Generate the depth map from a stereo image pair.

    Args:
        img_left (numpy.ndarray): the left stereo image.
        img_right (numpy.ndarray): the right stereo image.
        P0 (numpy.ndarray): the projection matrix of the left camera.
        P1 (numpy.ndarray): the projection matrix of the right camera.
    
    Returns:
        numpy.ndarray: the depth map.
    """

    # compute the disparity map
    disparity = compute_left_disparity_map(img_left, img_right)

    # decompose the projection matrices
    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)

    # calculate the depth map
    depth_map = calc_depth_map(disparity, k_left, t_left, t_right)

    return depth_map


def main(test_function):
    """
    Main function to test the utility functions.

    Args:
        test_function (str): the function to test.
    """

    if test_function == 'stereo_2_depth':
        from data import Dataset_Handler
        # test the stereo_2_depth function
        dataset = Dataset_Handler('00')
        depth_map = stereo_2_depth(dataset.first_image_left, dataset.first_image_right, dataset.P0, dataset.P1)
        plt.figure(figsize=(10, 10))
        plt.imshow(depth_map)
        plt.title('Depth map')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility functions to manipulate image frames.')
    parser.add_argument('--test_function', type=str,
                        default='stereo_2_depth',
                        choices=['stereo_2_depth'], 
                        help='The function to test.')

    args = parser.parse_args()

    main(args.test_function)


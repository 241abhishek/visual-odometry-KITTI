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
    matcher = cv2.StereoBM_create(numDisparities=96, blockSize=11)

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


def main(test_function):
    """
    Main function to test the utility functions.

    Args:
        test_function (str): the function to test.
    """

    if test_function == 'compute_left_disparity_map':
        # test the compute_left_disparity_map function
        img_left = cv2.imread('../dataset/sequences/00/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread('../dataset/sequences/00/image_1/000000.png', cv2.IMREAD_GRAYSCALE)
        disparity = compute_left_disparity_map(img_left, img_right, verbose=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(disparity)
        plt.title('Disparity map')
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    
    if test_function == 'decompose_projection_matrix':
        # test the decompose_projection_matrix function
        p = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                      [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                      [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])
        k, r, t = decompose_projection_matrix(p)
        print('Intrinsic matrix:')
        print(k)
        print('Rotation matrix:')
        print(r)
        print('Translation vector:')
        print(t)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility functions to manipulate image frames.')
    parser.add_argument('--test_function', type=str,
                        default='compute_left_disparity_map',
                        choices=['compute_left_disparity_map', 'decompose_projection_matrix'], 
                        help='The function to test.')

    args = parser.parse_args()

    main(args.test_function)


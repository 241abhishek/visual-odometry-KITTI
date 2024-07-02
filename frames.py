"""
Utility functions to manipulate image frames.
"""
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
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


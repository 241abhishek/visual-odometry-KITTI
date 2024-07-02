"""
Data handling and manipulation for visual odometry.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

class Dataset_Handler():
    """
    Class to handle data in the KITTI dataset.

    Attributes:
        seq_dir (str): the directory of the sequence.
        pose_dir (str): the directory of the ground truth poses.
        left_image_files (list): list of file names of the left images.
        right_image_files (list): list of file names of the right images.
        num_frames (int): number of frames in the sequence.
        P0 (numpy.ndarray): the projection matrix for the left grayscale camera.
        P1 (numpy.ndarray): the projection matrix for the right grayscale camera.
        P2 (numpy.ndarray): the projection matrix for the right rgb camera.
        P3 (numpy.ndarray): the projection matrix for the right rgb camera.
        times (numpy.ndarray): array of timestamps for each frame.
        gt (numpy.ndarray): array of ground truth poses for each frame.
        images_left (generator): generator to load left grayscale camera images sequentially.
        images_right (generator): generator to load right grayscale camera images sequentially.
        img_height (int): height of the images.
        img_width (int): width of the images.
        first_image_left (numpy.ndarray): the first left image in the sequence.
        first_image_right (numpy.ndarray): the first right image in the sequence.
        second_image_left (numpy.ndarray): the second left image in the sequence
    """    
    def __init__(self, sequence):
        """
        Initialize the dataset handler.

        Args:
            sequence (str): the sequence number of the dataset to load.
        """

        # set file paths and get ground truth poses
        self.seq_dir = os.path.join('../dataset/sequences', sequence)
        self.pose_dir = os.path.join('../dataset/poses', sequence + '.txt')
        poses = pd.read_csv(self.pose_dir, delimiter=' ', header=None)

        # get image file names
        self.left_image_files = sorted(os.listdir(os.path.join(self.seq_dir, 'image_0')), key = lambda x: int(x.split('.')[0]))
        self.right_image_files = sorted(os.listdir(os.path.join(self.seq_dir, 'image_1')), key = lambda x: int(x.split('.')[0]))
        self.num_frames = len(self.left_image_files)
        
        # fetch calibration parameters
        calib_file = os.path.join(self.seq_dir, 'calib.txt')
        calib = pd.read_csv(calib_file, delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape((3, 4))
        self.P1 = np.array(calib.loc['P1:']).reshape((3, 4))
        self.P2 = np.array(calib.loc['P2:']).reshape((3, 4))
        self.P3 = np.array(calib.loc['P3:']).reshape((3, 4))

        # fetch timestamps and ground truth poses
        self.times = np.array(pd.read_csv(os.path.join(self.seq_dir, 'times.txt'), delimiter=' ', header=None)) # timestamps
        self.gt = np.zeros((len(poses), 3, 4)) # ground truth poses
        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape((3,4))

        # use generators to load data seqeuntially to save memory
        self.reset_frames()

        # store some frames in memory for visualization and testing
        self.first_image_left = cv2.imread(os.path.join(self.seq_dir, 'image_0', self.left_image_files[0]), cv2.IMREAD_GRAYSCALE)
        self.first_image_right = cv2.imread(os.path.join(self.seq_dir, 'image_1', self.right_image_files[0]), cv2.IMREAD_GRAYSCALE)
        self.second_image_left = cv2.imread(os.path.join(self.seq_dir, 'image_0', self.left_image_files[1]), cv2.IMREAD_GRAYSCALE)

        # store image dimensions
        self.img_height, self.img_width = self.first_image_left.shape

    def reset_frames(self):
        """
        Reset the generators to the first frame.
        """
        # reset the generators to the first frame
        self.images_left = (cv2.imread(os.path.join(self.seq_dir, 'image_0', img), cv2.IMREAD_GRAYSCALE) for img in self.left_image_files)
        self.images_right = (cv2.imread(os.path.join(self.seq_dir, 'image_1', img), cv2.IMREAD_GRAYSCALE) for img in self.right_image_files)
        
if __name__ == '__main__':
    # test the Dataset_Handler class
    dataset = Dataset_Handler('00')

    print('Number of frames:', dataset.num_frames)
    print('Image dimensions:', dataset.img_height, dataset.img_width)
    print('Ground truth poses shape:', dataset.gt.shape)
    print('Ground truth pose at frame 0:', dataset.gt[0])

    # visualize the first frames
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.imshow(dataset.first_image_left, cmap='gray')
    plt.title('Left image')
    plt.subplot(212)
    plt.imshow(dataset.first_image_right, cmap='gray')
    plt.title('Right image')
    plt.show()
    plt.close()

    # test the generator, visualize 5 frames
    img_left = next(dataset.images_left)
    img_right = next(dataset.images_right)
    for i in range(5):
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.imshow(img_left, cmap='gray')
        plt.title('Left image frame ' + str(i+1))
        plt.subplot(212)
        plt.imshow(img_right, cmap='gray')
        plt.title('Right image frame ' + str(i+1))
        plt.show()
        for i in range(5):
            # skip 5 frames
            img_left = next(dataset.images_left)
            img_right = next(dataset.images_right)


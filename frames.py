"""
Utility functions to manipulate image frames.
"""
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse

def compute_left_disparity_map(img_left, img_right, verbose=False, matcher_name='bm'):
    """
    Compute the disparity map from a stereo image pair.

    Args:
        img_left (numpy.ndarray): the left stereo image.
        img_right (numpy.ndarray): the right stereo image.
        verbose (bool, optional): tag to toggle debug information. Defaults to False.
        matcher_name (str, optional): the name of the matcher to use. Defaults to 'bm'.

    Returns:
        numpy.ndarray: the disparity map.
    """

    # compute the disparity map

    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=96, blockSize=11)
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=96, minDisparity=0, blockSize=11,
                                        P1 = 8*1*11**2, P2 = 32*1*11**2,
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

def stereo_2_depth(img_left, img_right, P0, P1, matcher_name='bm'):
    """
    Generate the depth map from a stereo image pair.

    Args:
        img_left (numpy.ndarray): the left stereo image.
        img_right (numpy.ndarray): the right stereo image.
        P0 (numpy.ndarray): the projection matrix of the left camera.
        P1 (numpy.ndarray): the projection matrix of the right camera.
        matcher_name (str, optional): the name of the matcher to use. Defaults to 'bm'.
    
    Returns:
        numpy.ndarray: the disparity map.
        numpy.ndarray: the depth map.
    """

    # compute the disparity map
    disparity = compute_left_disparity_map(img_left, img_right, matcher_name=matcher_name)

    # decompose the projection matrices
    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)

    # calculate the depth map
    depth_map = calc_depth_map(disparity, k_left, t_left, t_right)

    return disparity, depth_map

def extract_features(img, mask=None):
    """
    Find keypoints and descriptors in an image.

    Args:
        img (numpy.ndarray): the image to extract features from.
        mask (numpy.ndarray, optional): the mask to apply to the image. Defaults to None.
    
    Returns:
        kp (list): the keypoints.
        des (list): the descriptors.
    """

    # create a SIFT detector object
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors
    kp, des = sift.detectAndCompute(img, mask)

    return kp, des

def match_features(des1, des2):
    """
    Match features between two sets of descriptors.

    By default, this function uses the brute force, sorted matching, and returns 2 nearest neighbors.

    Args:
        des1 (list): the first set of descriptors.
        des2 (list): the second set of descriptors.
    
    Returns:
        matches (list): the matched features.
    """

    # create a BFMatcher object and match the features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # sort the matches based on distance
    matches = sorted(matches, key = lambda x:x[0].distance)

    return matches

def filter_matches_distance(matches, threshold=0.5):
    """
    Filter matches based on the distance ratio.

    Args:
        matches (list): the matched features.
        threshold (float, optional): the distance ratio threshold. Defaults to 0.75.
    
    Returns:
        filtered_matches (list): the filtered matches.
    """

    # filter matches based on the distance ratio
    filtered_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            filtered_matches.append(m)

    return filtered_matches

def visualize_matches(img1, img2, kp1, kp2, matches):
    """
    Visualize the matched features between two images.

    Args:
        img1 (numpy.ndarray): the first image.
        img2 (numpy.ndarray): the second image.
        kp1 (list): the keypoints of the first image.
        kp2 (list): the keypoints of the second image.
        matches (list): the matched features.
    """

    # draw the matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # display the matches
    plt.figure(figsize=(10, 10))
    plt.imshow(img_matches)
    plt.title('Matches')
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def estimate_motion(match, kp1, kp2, k, depth_map, max_depth=3000):
    """
    Estimate the motion of the camera between two frames.

    Args:
        match (list): the matched feature.
        kp1 (list): the keypoints of the first image.
        kp2 (list): the keypoints of the second image.
        k (numpy.ndarray): the intrinsic matrix.
        depth_map (numpy.ndarray): the depth map.
        max_depth (int, optional): the maximum depth. Defaults to 3000.

    Returns:
        numpy.ndarray: the rotation matrix.
        numpy.ndarray: the translation vector.
        numpy.ndarray: matched feature coordinates (u,v pixels) in the first image.
        numpy.ndarray: matched feature coordinates (u,v pixels) in the second image.
    """

    rmat = np.eye(3)
    tvec = np.zeros((3, 1))

    # get the coordinates of the matched features
    image_points_1 = np.float32([kp1[m.queryIdx].pt for m in match])
    image_points_2 = np.float32([kp2[m.trainIdx].pt for m in match])

    # extract the intrinsic parameters
    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]

    # calculate the 3D points
    object_points = np.zeros((0, 3))
    delete = []

    for i, (u,v) in enumerate(image_points_1):
        # get the depth
        # flipped u and v because of the row, column convention in numpy
        z = depth_map[int(v), int(u)]
        
        if z > max_depth:
            delete.append(i)
            continue

        # calculate the 3D point
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        object_points = np.vstack((object_points, np.array([x, y, z])))

    # remove points with depth greater than the maximum depth
    image_points_1 = np.delete(image_points_1, delete, axis=0)
    image_points_2 = np.delete(image_points_2, delete, axis=0)

    # estimate the motion using the PnP RANSAC algorithm
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points_2, k, None)

    # convert the rotation vector to a rotation matrix
    rmat = cv2.Rodrigues(rvec)[0]

    return rmat, tvec, image_points_1, image_points_2

def visual_odometry(handler, matcher_name='sgbm', filter_match_distance=0.3, subset=None, plot=False, visualize=False):
    """
    The full visual odometry pipeline.
    
    Args:
        handler (Dataset_Handler): the dataset handler.
        matcher_name (str, optional): the name of the matcher to use. Defaults to 'sgbm'.
        filter_match_distance (float, optional): the distance ratio threshold. Defaults to 0.3.
        subset (int, optional): the number of frames to process. Defaults to None.
        plot (bool, optional): toggle 2D plot. Defaults to False.
        visualize (bool, optional): toggle visualization of images. Defaults to False.

    Returns:
        numpy.ndarray: the ground truth poses.
        numpy.ndarray: the estimated poses.
    """

    # initialize the number of frames to process
    if subset is not None:
        num_frames = subset
    else:
        num_frames = handler.num_frames

    if plot and visualize:
        visualize = False

    if plot:
        # construct a 2d plot excluding the z axis
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        xs = handler.gt[:, 0, 3]
        ys = handler.gt[:, 1, 3]
        ax.plot(xs, ys, c='k')
        # set title and labels
        ax.set_title('2D path visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(['Ground truth'])

    if visualize:
        # construct a 2d plot excluding the z axis
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(311)
        xs = handler.gt[:, 0, 3]
        ys = handler.gt[:, 1, 3]
        ax.plot(xs, ys, c='k')
        # set title and labels
        ax.set_title('2D path visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(['Ground truth'])

    # initialize the poses
    estimated_trajectory = np.zeros((num_frames, 3, 4))
    t_tot = np.eye(4)
    estimated_trajectory[0] = t_tot[:3, :]

    # decompose the projection matrices
    k_left, r_left, t_left = decompose_projection_matrix(handler.P0)

    # reset the frames 
    handler.reset_frames()
    image_plus1 = next(handler.images_left)

    # iterate through all the frames in the sequence
    # set the range to num_frames - 1 to avoid index out of bounds
    # when processing 2 sequential frames
    for i in range(num_frames - 1):
        image_left = image_plus1
        image_right = next(handler.images_right)

        if visualize:
            # display the left camera image
            ax_1 = fig.add_subplot(312)
            ax_1.imshow(image_left, cmap='gray')
            ax_1.set_title('Grayscale Camera Image')

        # get the next left camera iamge for visual odometry
        image_plus1 = next(handler.images_left)

        disp, depth_map = stereo_2_depth(image_left, image_right, handler.P0, handler.P1, matcher_name=matcher_name)

        # display the disparity map
        if visualize:
            ax_2 = fig.add_subplot(313)
            # crop the disp map to remove non-overlapping regions
            ax_2.imshow(disp[:, 96:]) # 96 is empirically determined
            ax_2.set_title('Disparity map')

        # display the depth map
        # if visualize:
        #     ax_2 = fig.add_subplot(313)
        #     ax_2.imshow(depth_map)
        #     ax_2.set_title('Depth map')

        # extract features from the images
        kp1, des1 = extract_features(image_left)
        kp2, des2 = extract_features(image_plus1)

        # display the feature dectected in the first image
        # if visualize and i%10 == 0:
        #     ax_2 = fig.add_subplot(313)
        #     kp_image = cv2.drawKeypoints(image_left, kp1, None)
        #     ax_2.imshow(kp_image)
        #     ax_2.set_title('Detected features')

        # match the features
        matches = match_features(des1, des2)

        # filter the matches
        matches = filter_matches_distance(matches, threshold=filter_match_distance)

        # estimate the motion
        rmat, tvec, image_points_1, image_points_2 = estimate_motion(matches, kp1, kp2, k_left, depth_map)

        # update the trajectory
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.flatten()
        t_tot = t_tot.dot(np.linalg.inv(Tmat))

        estimated_trajectory[i + 1] = t_tot[:3, :]

        if plot:
            # plot the 2D path
            xs = estimated_trajectory[:i + 2, 0, 3]
            ys = estimated_trajectory[:i + 2, 1, 3]
            ax.plot(xs, ys, c='chartreuse')
            plt.pause(1e-32)
            ax.legend(['Ground truth', 'Estimated'])

        if visualize:
            # plot the 2D path
            xs = estimated_trajectory[:i + 2, 0, 3]
            ys = estimated_trajectory[:i + 2, 1, 3]
            ax.plot(xs, ys, c='chartreuse')
            plt.pause(1e-32)
            ax.legend(['Ground truth', 'Estimated'])

    if plot or visualize:
        plt.close()

    return handler.gt, estimated_trajectory

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
        _, depth_map = stereo_2_depth(dataset.first_image_left, dataset.first_image_right, dataset.P0, dataset.P1, matcher_name='sgbm')
        plt.figure(figsize=(10, 10))
        plt.imshow(depth_map)
        plt.title('Depth map')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    if test_function == 'visualize_matches':
        from data import Dataset_Handler
        # test the visualize_matches function
        dataset = Dataset_Handler('00')
        img_1 = dataset.first_image_left
        img_2 = dataset.second_image_left
        kp_1, des_1 = extract_features(img_1)
        kp_2, des_2 = extract_features(img_2)
        matches = match_features(des_1, des_2)
        print('Number of matches before filtering:', len(matches))
        matches = filter_matches_distance(matches, threshold=0.3)
        print('Number of matches after filtering:', len(matches))
        visualize_matches(img_1, img_2, kp_1, kp_2, matches)

    if test_function == 'estimate_motion':
        from data import Dataset_Handler
        # test the estimate_motion function
        dataset = Dataset_Handler('00')
        img_1 = dataset.first_image_left
        img_2 = dataset.second_image_left
        img_r = dataset.first_image_right
        kp_1, des_1 = extract_features(img_1)
        kp_2, des_2 = extract_features(img_2)
        matches = match_features(des_1, des_2)
        _, depth_map = stereo_2_depth(img_1, img_r, dataset.P0, dataset.P1, matcher_name='sgbm')
        matches = filter_matches_distance(matches, threshold=0.3)
        rmat, tvec, image_points_1, image_points_2 = estimate_motion(matches, kp_1, kp_2, decompose_projection_matrix(dataset.P0)[0], depth_map)
        print('Rotation matrix:', rmat.round(3))
        print('Translation vector:', tvec.round(3))

    if test_function == 'visual_odometry':
        from data import Dataset_Handler
        # test the visual_odometry function
        dataset = Dataset_Handler('00')
        trajectory = visual_odometry(dataset, matcher_name='sgbm', filter_match_distance=0.3, subset=100, plot=True)
        print('Estimated poses shape:', trajectory.shape)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility functions to manipulate image frames.')
    parser.add_argument('--test_function', type=str,
                        default='stereo_2_depth',
                        choices=['stereo_2_depth', 'visualize_matches', 'estimate_motion', 'visual_odometry'], 
                        help='The function to test.')

    args = parser.parse_args()

    main(args.test_function)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_3d_path(poses):
    """
    Visualize the path of the camera in 3D space.

    Args:
        poses (numpy.ndarray): a numpy array of poses, where each pose is a 3x4 matrix.
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poses[:,0,3], poses[:,1,3], poses[:,2,3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D path visualization')
    ax.view_init(elev=-40, azim=270)
    plt.show()


if __name__ == '__main__':
    # test the visualize_3d_path function
    poses = pd.read_csv('../dataset/poses/00.txt', delimiter=' ', header=None)
    gt = np.zeros((poses.shape[0], 3, 4))
    for i in range(poses.shape[0]):
        gt[i] = np.array(poses.iloc[i]).reshape(3, 4)
    visualize_3d_path(gt)

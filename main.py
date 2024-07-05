"""
Script to run the viusal oometry pipeline on the KITTI dataset
and visualize the results

"""

from data import Dataset_Handler
from frames import visual_odometry
from utils import visualize_3d_path

def main():
    # load the dataset
    dataset = Dataset_Handler('00')
    print('Number of frames:', dataset.num_frames)
    print('Image dimensions:', dataset.img_height, dataset.img_width)
    
    # run visual odometry
    poses = visual_odometry(dataset, plot=True)

    # # visualize the 3D path
    # dataset.visualize_3d_path(poses)

if __name__ == "__main__":
    main()
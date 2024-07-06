"""
Script to run the viusal oometry pipeline on the KITTI dataset
and visualize the results

"""

from data import Dataset_Handler
from frames import visual_odometry
from utils import visualize_3d_path
from argparse import ArgumentParser

def main(*args):

    args = ArgumentParser()
    args.add_argument('--sequence', type=str, default='00', help='KITTI sequence number - from 00 to 10')
    args = args.parse_args()

    # load the dataset
    seq = args.sequence
    dataset = Dataset_Handler(seq)
    print('Number of frames:', dataset.num_frames)
    print('Image dimensions:', dataset.img_height, dataset.img_width)
    
    # run visual odometry
    ground_truth, estimated_trajectory = visual_odometry(dataset, plot=True, visualize=False, subset=100, save=True)

    print('Ground truth shape:', ground_truth.shape)
    print('Estimated trajectory shape:', estimated_trajectory.shape)

if __name__ == "__main__":
    main()
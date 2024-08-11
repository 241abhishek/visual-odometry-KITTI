# Visual Odometry using KITTI Dataset

This project implements a visual odometry pipeline using stereo images from the KITTI dataset. The system estimates the trajectory of a moving vehicle by analyzing the movement of visual features across consecutive frames.

## Features

- Stereo depth estimation
- Feature extraction and matching
- Motion estimation using PnP and RANSAC
- Visual odometry pipeline
- 2D and 3D trajectory visualization
- Animation creation from image sequences
- Flexible dataset handling for KITTI sequences

## Project Structure

- `main.py`: Main script to run the visual odometry pipeline
- `data.py`: Handles data loading and preprocessing for KITTI dataset
- `frames.py`: Contains core functions for visual odometry processing
- `utils.py`: Utility functions (including 3D path visualization)

## Pipeline Overview

1. **Data Loading**: 
   - Load KITTI sequence data using `Dataset_Handler`
   - Access calibration parameters, ground truth poses, and image sequences

2. **Stereo Depth Estimation**: 
   - Compute disparity map using either StereoBM or StereoSGBM
   - Calculate depth map from disparity

3. **Feature Handling**:
   - Extract SIFT features from consecutive frames
   - Match features using Brute Force matcher
   - Filter matches based on distance ratio

4. **Motion Estimation**:
   - Use PnP (Perspective-n-Point) with RANSAC for robust estimation
   - Estimate rotation and translation between frames

5. **Trajectory Reconstruction**:
   - Accumulate frame-to-frame transformations
   - Reconstruct full trajectory

6. **Visualization**:
   - 2D plot of estimated trajectory vs ground truth
   - Optional 3D path visualization
   - Optional visualization of intermediate results (disparity, depth, features, matches)

## Key Components

### Dataset Handler (`data.py`)

- `Dataset_Handler` class:
  - Loads and manages KITTI sequence data
  - Provides access to calibration parameters, ground truth poses, and image sequences
  - Uses generators for memory-efficient image loading

### Visual Odometry Pipeline (`frames.py`)

- `stereo_2_depth`: Generates depth map from stereo image pair
- `extract_features`: Finds keypoints and descriptors in an image
- `match_features`: Matches features between two sets of descriptors
- `estimate_motion`: Estimates camera motion between two frames
- `visual_odometry`: Implements the full visual odometry pipeline
- `create_animation`: Creates animations from image sequences

### Main Script (`main.py`)

- Parses command-line arguments for KITTI sequence selection
- Initializes `Dataset_Handler`
- Runs the visual odometry pipeline
- Optionally saves results and creates visualizations

## Usage

To run the visual odometry pipeline:

```
python main.py --sequence 00
```
You can replace `00` with any valid KITTI sequence number (00 to 10).

Ensure you have the KITTI dataset downloaded and extracted in a directory named `dataset` at the parent level of this repository before running the code.

## Results

The system generates:
- Estimated trajectory plot 
- Animations of:
  - 2D path
  - Grayscale camera images
  - Disparity maps
  - Detected features
  - Depth maps
  - Matched features

## Dependencies

- OpenCV
- NumPy
- Matplotlib
- Pandas
- tqdm

## Future Improvements

- Implement loop closure detection
- Integrate with mapping algorithms for full SLAM
- Extend support for other datasets
- Optimize for real-time performance

## Acknowledgments

- KITTI Dataset for providing the stereo image sequences and ground truth
- OpenCV community for computer vision tools and algorithms
- This project was inspired by the tutorial series: [Visual Odometry for Beginners](https://youtube.com/playlist?list=PLrHDCRerOaI9HfgZDbiEncG5dx7S3Nz6X&si=MDN_iUHuaNn3Nmyu)
- The accompanying GitHub repository: [KITTI_visual_odometry](https://github.com/FoamoftheSea/KITTI_visual_odometry) by FoamoftheSea
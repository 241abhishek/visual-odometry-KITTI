# Visual Odometry using KITTI Dataset

This project implements a visual odometry pipeline using stereo images from the KITTI dataset. The system estimates the trajectory of a moving vehicle by analyzing the movement of visual features across consecutive frames.

## Demo Video

<div align="center">
  <video src= https://private-user-images.githubusercontent.com/72541517/356891592-06eebce6-0d7a-43f7-bf2b-d5071d9a038b.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjMzOTM3OTUsIm5iZiI6MTcyMzM5MzQ5NSwicGF0aCI6Ii83MjU0MTUxNy8zNTY4OTE1OTItMDZlZWJjZTYtMGQ3YS00M2Y3LWJmMmItZDUwNzFkOWEwMzhiLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA4MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwODExVDE2MjQ1NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTgzYzc4ZDc4YjUzNjMyZDQ1ZGQ4YTc1ZjRmMjVlN2JjZWNmOWExN2VjNDJlNGEyNWU3MGVkMGEyNTdmNzgzMTImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.n6ub4JGWSqtnhoQxOvKQr5Gey8nZ8C3dg3YYYHaiFsQ />
</div>

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
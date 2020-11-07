# Uncertainty-Aware Fast Curb Detection Using\\ Convolutional Networks in Point Clouds
## **Introduction**

## Data set
```
├── Total_data_set_line
│   ├── Training
│   │   ├── input_data
│   │   │             ├──cue_points 
│   │   │             ├──density_map
│   │   │             └──slice_map
│   │   ├── labels
│   │   └── top_view_raw   
│   └── test
│       └── labels            
│
└── Total_data_set_point
│   ├── Training
│   │   ├── input_data
│   │   │             ├──cue_points 
│   │   │             ├──density_map
│   │   │             └──slice_map                      
│   │   ├── labels
│   │   └── top_view_raw   
│   └── test
│        └── labels
```
Folder descrption
* Total_data_set_line : The label format is lines in 2D image domain.
* Total_data_set_point : The label format is points in 2D image domain.
* input_data : It consists of three folders; cue_points, density_map, and slice_map
* labels : 2D image(320x416). The pixel values for curb area are one. The others area are zero.
* top_view_raw : The bird's eye view image of 3D point cloud for visualization.
* test : It contains the only lables for test phase. The inputs are included in training folder. You can use the file name of labels in the test foler to extract the input data.

The link for dataset is


## Project Code 
TDB
### Requirements
* Python 3.5
* Tensorflow 1.11

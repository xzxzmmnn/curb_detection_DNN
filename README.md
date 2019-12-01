# Fast Curb Detection based on a Single-Shot Point Cloud with Cue Points using Deep Neural Networks
## **Introduction**

Younghwa Jung, Mingu Jeon, Chan Kim, Seung-Woo Seo, and Seong-Woo Kim, "Fast Curb Detection based on a Single-Shot Point Cloud with Cue Points Using Deep Neural Networks," submitted to International Conference on Robotics and Automation, 2020.
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
** what
* labels : 2D image(320x416). The pixel values for curb area are one. The others area are zero.
* top_view_raw : The bird's eye view image of 3D point cloud for visualization.


The link for dataset is https://drive.google.com/drive/folders/1RuApVar81cC59NZgJmkFyBwkfGUYK79o?usp=sharing


## Project Code 
TDB
### Requirements
* Python 3.5
* Tensorflow 1.11

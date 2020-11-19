# Uncertainty-Aware Fast Curb Detection Using\\ Convolutional Networks in Point Clouds
## **Introduction**
Younghwa Jung, Mingu Jeon, Chan Kim, Seung-Woo Seo, and Seong-Woo Kim, "Uncertainty-Aware Fast Curb Detection Using Convolutional Networks in Point Clouds," submitted to IEEE International Conference on Robotics and Automation (ICRA), 2021
## Data set
```
├── Total_data_set_line
│   ├── Training
│   │   ├── input_data
│   │   │             ├──density_map 
│   │   │             └──heights_maps
│   │   │             
│   │   ├── labels
│   │   └── top_view_raw   
│   └── test
│       └── labels            
│
└── Total_data_set_point
│   ├── Training
│   │   ├── input_data
│   │   │             ├──density_map 
│   │   │             └──heights_maps
│   │   │                                   
│   │   ├── labels
│   │   └── top_view_raw   
│   └── test
│        └── labels
```
Folder descrption
* Total_data_set_line : The label format is lines in 2D image domain.
* Total_data_set_point : The label format is points in 2D image domain.
* input_data : It consists of three folders; density_map and height_maps
* labels : 2D image(320x416). The pixel values for curb area are one. The others area are zero.
* top_view_raw : The bird's eye view image of 3D point cloud for visualization.
* test : It contains the only lables for test phase. The inputs are included in training folder. You can use the file name of labels in the test foler to extract the input data.

The link for dataset is

## Label Extraction from Semantic Map 
[![IMAGE ALT TEXT HERE](https://drive.google.com/uc?export=view&id=1R-ljnWRGG1t4iyzx7pkNQYeYgu1JkH-5)](https://youtu.be/2d28cw9zb-0)


## Project Code 
TDB
### Requirements
* Python 3.5
* Tensorflow 1.11

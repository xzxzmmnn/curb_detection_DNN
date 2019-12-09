from struct import Struct
import numpy as np
import cv2
import math
import os
from scipy import misc
import tensorflow as tf
import random
from datetime import datetime
import Data.config as cfg
import glob

class CommonFunc:
    def __init__(self):
        pass


    def shuffle_batch_data(self,step,number_set,train_data,label_data,batch_size=5):
        batch_number=number_set[(step-1)*batch_size:step*batch_size]
        batch_data=[]
        batch_label=[]
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_label.append(label_data[idx])


        loaded_label = np.concatenate(batch_label, axis=0).reshape(-1, 400, 300, 1)
        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, 400, 300, 3)

        return loaded_data, loaded_label



    def bd_data_gen_6(self,data_path):
        file_list=glob.glob(data_path+"boundary_image/*.png")
        data_list, bd_label_list, frame_num_list=[],[],[]
        for file_name in file_list:
            frame_number = file_name.split('/')[-1].split('.')[0]
            data = self.get_slice_data_3(frame_number,data_path)#density map, slice height map(5)
            bd_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            data_list.append(data)
            bd_label_list.append(bd_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 6)
        bd_batch = np.concatenate(bd_label_list, axis=0).reshape(-1, 416, 320, 1)

        return data_batch, bd_batch, frame_num_list

    
    
    def test_data_gen_2(self,data_path,root_path):
        file_list=glob.glob(data_path+"boundary_image/*.png")
        data_list, bd_label_list, frame_num_list=[],[],[]
        for file_name in file_list:
            frame_number = file_name.split('/')[-1].split('.')[0]
            data = self.get_test_slice_data_2(frame_number,root_path)
            bd_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            data_list.append(data)
            bd_label_list.append(bd_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 6)
        bd_batch = np.concatenate(bd_label_list, axis=0).reshape(-1, 416, 320, 1)

        return data_batch, bd_batch, frame_num_list    
        

    def get_test_slice_data_2(self, frame_number,root_path):#using visual cue data.
        density_path=root_path+"input_data/density_map/"+str(frame_number)+".png"
        density_image=cv2.imread(density_path,cv2.IMREAD_GRAYSCALE)
        visual_cue_path=root_path+"input_data/visual_cue/"+str(frame_number)+".png"
        visual_cue_image=cv2.imread(visual_cue_path,cv2.IMREAD_GRAYSCALE)
        current_slice_set=[]
        for i in range(1,5):
            slice_path=root_path+"input_data/slice_map/"+str(frame_number)+"_"+str(i)+".png"
            slice_image=cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            current_slice_set.append(slice_image)

        current_slice_set.append(density_image)
        current_slice_set.append(visual_cue_image)
        total_data_set=np.concatenate(current_slice_set,axis=0).reshape(416,320,6)
        return total_data_set    

    def get_slice_data_3(self, frame_number,root_path):#adding visual cue
        density_path=root_path+"input_data/density_map/"+str(frame_number)+".png"
        density_image=cv2.imread(density_path,cv2.IMREAD_GRAYSCALE)
        visual_cue_path=root_path+"input_data/visual_cue/"+str(frame_number)+".png"
        visual_cue_image=cv2.imread(visual_cue_path,cv2.IMREAD_GRAYSCALE)

        current_slice_set=[]
        for i in range(1,5):
            slice_path=root_path+"input_data/slice_map/"+str(frame_number)+"_"+str(i)+".png"
            slice_image=cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            current_slice_set.append(slice_image)

        current_slice_set.append(density_image)
        current_slice_set.append(visual_cue_image)
        total_data_set=np.concatenate(current_slice_set,axis=0).reshape(416,320,6)
        return total_data_set




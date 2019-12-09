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

    def readkitti_vel(self,path):
        scan=np.fromfile(path,dtype=np.float32)
        return scan.reshape((-1,4))

    def converXYZRGB(self,point):
        rho=math.sqrt(point[0]*point[0]+point[1]*point[1]+point[2]*point[2])
        theta=math.atan2(point[1],point[0])/math.pi*(180)
        phi=math.atan2(math.sqrt(point[0]*point[0]+point[1]*point[1]),point[2])
        return rho, theta, phi

    def load_label_kitti(selfs,path):
        img=np.zeros((400,300,3),np.float32)


    def pcd_to_img(self,pointCloud,gridsize):
        img = np.zeros((400, 300), np.uint8)
        y_bias=150
        for i in range(pointCloud.shape[0]):
            if(pointCloud[i,0]>=0 and pointCloud[i,0]<=40 and pointCloud[i,1]>=-15 and pointCloud[i,1]<=15):
                idx_x=int(pointCloud[i,0]/gridsize) if pointCloud[i,0]>=0 else int(pointCloud[i,0]/gridsize)-1
                idx_y=int(pointCloud[i,1]/gridsize) if pointCloud[i,1]>=0 else int(pointCloud[i,1]/gridsize)-1
                idx_y+=y_bias

                if(idx_x>=0 and idx_x<=400 and idx_y>=0 and idx_y<300):
                    img[idx_x, idx_y]=255
        flipped_img = cv2.flip(img, -1)
        return flipped_img

    def pcd_to_img_2(self,pointCloud,gridsize):
        img=np.zeros((400,300),np.float32)
        y_bias=150
        for i in range(pointCloud.shape[0]):
            if (pointCloud[i, 0] >= 0 and pointCloud[i, 0] <= 40 and pointCloud[i, 1] >= -15 and pointCloud[i, 1] <= 15):
                idx_x = int(pointCloud[i, 0] / gridsize) if pointCloud[i, 0] >= 0 else int(
                    pointCloud[i, 0] / gridsize) - 1
                idx_y = int(pointCloud[i, 1] / gridsize) if pointCloud[i, 1] >= 0 else int(
                    pointCloud[i, 1] / gridsize) - 1
                idx_y += y_bias

                if (idx_x >= 0 and idx_x <= 400 and idx_y >= 0 and idx_y < 300):
                    img[idx_x, idx_y] = 255

        flipped_img = cv2.flip(img, -1)
        return flipped_img



    def pcd_to_img_3(self,pointCloud,gridsize):
        img = np.zeros((400, 300,3), np.float32)
        y_bias=150
        for i in range(pointCloud.shape[0]):
            if(pointCloud[i,0]>=0 and pointCloud[i,0]<=40 and pointCloud[i,1]>=-15 and pointCloud[i,1]<=15):
                idx_x=int(pointCloud[i,0]/gridsize) if pointCloud[i,0]>=0 else int(pointCloud[i,0]/gridsize)-1
                idx_y=int(pointCloud[i,1]/gridsize) if pointCloud[i,1]>=0 else int(pointCloud[i,1]/gridsize)-1
                idx_y+=y_bias

                if(idx_x>=0 and idx_x<=400 and idx_y>=0 and idx_y<300):
                    img[idx_x, idx_y, 0] = 255
                    img[idx_x, idx_y, 1] = 255
                    img[idx_x, idx_y, 2] = 255
        flipped_img = cv2.flip(img, -1)
        return flipped_img

    def pcd_to_img_4(self,vel_path,gridsize):
        img = np.zeros((400, 300,3), np.float32)
        y_bias=150
        pointCloud = self.readkitti_vel(vel_path)
        for i in range(pointCloud.shape[0]):
            if (pointCloud[i, 0] >= 0 and pointCloud[i, 0] < 40 and pointCloud[i, 1] >= -15 and pointCloud[i, 1] < 15 and pointCloud[i, 2] >= -2 and pointCloud[i, 2] <= 1.25):
                idx_x = int(pointCloud[i, 0] / gridsize) if pointCloud[i, 0] >= 0 else int(pointCloud[i, 0] / gridsize) - 1
                idx_y = int(pointCloud[i, 1] / gridsize) if pointCloud[i, 1] >= 0 else int(pointCloud[i, 1] / gridsize) - 1
                idx_y += y_bias
                if (idx_x >= 0 and idx_x < 400 and idx_y >= 0 and idx_y < 300):
                    img[idx_x, idx_y, 0] = 255
                    img[idx_x, idx_y, 1] = 255
                    img[idx_x, idx_y, 2] = 255

        flipped_img = cv2.flip(img, -1)
        return flipped_img


    def top_view_with_label(self,top_view,label):
        for row in range(top_view.shape[0]):
            for col in range(top_view.shape[1]):
                if(label[row,col,2]==255):
                    top_view[row,col, 0] = 0
                    top_view[row, col,1] = 0
                    top_view[row, col,2] = 1


        return top_view


    def moving_mean_variance(self,data,count,pwrSumAvg,mean,variance):
        mean += ((data-mean)/count)
        pwrSumAvg += (data*data-pwrSumAvg)/count
        if(count>=2):
            variance=math.sqrt((pwrSumAvg*count-count*mean*mean)/(count-1))
        return mean,pwrSumAvg,variance

    def moving_mean(self,data,count,mean):
        mean+=(data-mean)/count
        return mean


    def feature_extraction_lidar(self, path,gridsize):#based on complex_yolo
        fea_map=np.full((400,300,4),-999,np.float32)
        y_bias=150

        pointCloud=self.readkitti_vel(path)
        for i in range(pointCloud.shape[0]):
            if(pointCloud[i,0]>=0 and pointCloud[i,0]<=40 and pointCloud[i,1]>=-15 and pointCloud[i,1]<=15 and pointCloud[i,2]>=-2 and pointCloud[i,2]<=1.25):
                idx_x = int(pointCloud[i, 0] / gridsize) if pointCloud[i, 0] >= 0 else int(pointCloud[i, 0] / gridsize) - 1
                idx_y = int(pointCloud[i, 1] / gridsize) if pointCloud[i, 1] >= 0 else int(pointCloud[i, 1] / gridsize) - 1
                idx_y += y_bias

                if(fea_map[idx_x,idx_y,0]<pointCloud[i, 2]):
                    fea_map[idx_x,idx_y,0]=pointCloud[i, 2]#for max height
                if(fea_map[idx_x,idx_y,1]<pointCloud[i,3]):
                    fea_map[idx_x, idx_y, 1] = pointCloud[i, 3]#for max intensity
                if(fea_map[idx_x, idx_y, 3]==-999):#first count for density
                    fea_map[idx_x, idx_y, 3] += 1000 #for density
                else:
                    fea_map[idx_x, idx_y, 3] += 1
                fea_map[idx_x,idx_y,2]+=min(1,math.log10(fea_map[idx_x, idx_y, 3])/64)

        fea_map=fea_map[:,:,0:3]
        flipped_fea = cv2.flip(fea_map, -1)


        return flipped_fea

    def feature_extraction_lidar_2(self, path,gridsize):#based on complex_yolo
        fea_map=np.full((400,300,3),0,np.float32)
        y_bias=150

        pointCloud=self.readkitti_vel(path)
        for i in range(pointCloud.shape[0]):
            if(pointCloud[i,0]>=0 and pointCloud[i,0]<40 and pointCloud[i,1]>=-15 and pointCloud[i,1]<15 and pointCloud[i,2]>=-2 and pointCloud[i,2]<=1.25):
                idx_x = int(pointCloud[i, 0] / gridsize) if pointCloud[i, 0] >= 0 else int(pointCloud[i, 0] / gridsize) - 1
                idx_y = int(pointCloud[i, 1] / gridsize) if pointCloud[i, 1] >= 0 else int(pointCloud[i, 1] / gridsize) - 1
                idx_y += y_bias
                if(idx_x>=0 and idx_x<400 and idx_y>=0 and idx_y<300):
                    fea_map[idx_x, idx_y, 0] += 1  # for density
                    if(fea_map[idx_x,idx_y,1]<pointCloud[i, 2]+2):
                        fea_map[idx_x,idx_y,1]=pointCloud[i, 2]+2#for max height
                    if(fea_map[idx_x,idx_y,2]<pointCloud[i,3]):
                        fea_map[idx_x, idx_y, 2] = pointCloud[i, 3]#for max intensity

        fea_map[:, :, 1]=fea_map[:, :, 1]/(3.25)#normalize for max_height
        fea_map[:, :, 2]=np.minimum(1,np.log(fea_map[:, :, 2]+1)/64)#normalize for vlp64
        flipped_fea = cv2.flip(fea_map, -1)

        return flipped_fea


    def batch_data(self,data_dir,label_dir):
        filename_list=os.listdir(data_dir)#numb of files
        file_size=len(filename_list)
        rand_idx=np.arange(file_size)
        np.random.shuffle(rand_idx)
        label_batch, data_batch=[],[]
        for frame_num in rand_idx:
            top_view_path = "{!s}/{:d}.png".format(data_dir, frame_num)
            label_path = "{!s}/{:d}.png".format(label_dir, frame_num)
            label_info = misc.imread(label_path)
            top_view_data = misc.imread(top_view_path)
            label_batch.append(label_info)
            data_batch.append(top_view_data)

        label_batch=np.concatenate(label_batch,axis=0).reshape(-1,400,300,1)
        data_batch=np.concatenate(data_batch,axis=0).reshape(-1,400,300,3)

        return data_batch,label_batch

    def batch_data_bin(self, data_dir, label_dir):
        filename_list = os.listdir(data_dir)  # numb of files
        file_size = len(filename_list)
        rand_idx = np.arange(file_size)
        np.random.shuffle(rand_idx)
        label_batch, data_batch = [], []
        for frame_num in rand_idx:
            top_view_path = "{!s}/{:d}.bin".format(data_dir, frame_num)
            label_path = "{!s}/{:d}.png".format(label_dir, frame_num)
            label_info = misc.imread(label_path)
            top_view_data = self.readstruct_4(top_view_path)
            label_batch.append(label_info)
            data_batch.append(top_view_data)

        label_batch = np.concatenate(label_batch, axis=0).reshape(-1, 400, 300, 1)
        data_batch = np.concatenate(data_batch, axis=0).reshape(-1, 400, 300, 3)

        return data_batch, label_batch

    def batch_data_bin_2(self, data_dir, label_dir):#because of orientation
        filename_list = os.listdir(data_dir)  # numb of files
        file_size = len(filename_list)
        rand_idx = np.arange(file_size)
        np.random.shuffle(rand_idx)
        label_batch, data_batch = [], []
        for frame_num in rand_idx:
            top_view_path = "{!s}/{:d}.bin".format(data_dir, frame_num)
            label_path = "{!s}/{:d}.png".format(label_dir, frame_num)
            label_info = misc.imread(label_path)
            top_view_data = self.readstruct_4(top_view_path)
            label_batch.append(label_info)
            data_batch.append(top_view_data)

        label_batch = np.concatenate(label_batch, axis=0).reshape(-1, 400, 300, 1)
        data_batch = np.concatenate(data_batch, axis=0).reshape(-1, 400, 300, 3)

        return data_batch,label_batch

    def batch_data_bin_3(self, data_dir, label_dir):  #y2do area 는숫자가연소적이지않음.
        filename_list = os.listdir(label_dir)  # numb of files
        frame_num_list=self.get_frame_number(filename_list)
        np.random.shuffle(frame_num_list)
        label_batch, data_batch = [], []
        for frame_num in frame_num_list:
            top_view_path = "{!s}/{:d}.bin".format(data_dir, frame_num)
            label_path = "{!s}/{:d}.png".format(label_dir, frame_num)
            label_info = misc.imread(label_path)
            top_view_data = self.readstruct_4(top_view_path)
            label_batch.append(label_info)
            data_batch.append(top_view_data)

        label_batch = np.concatenate(label_batch, axis=0).reshape(-1, 400, 300, 1)
        data_batch = np.concatenate(data_batch, axis=0).reshape(-1, 400, 300, 3)

        return data_batch,label_batch

    def batch_data_bin_4(self, data_dir, label_dir):  #y2do area 는숫자가연소적이지않음.
        filename_list = os.listdir(label_dir)  # numb of files
        frame_num_list=self.get_frame_number(filename_list)
        np.random.shuffle(frame_num_list)
        label_batch, data_batch = [], []
        for frame_num in frame_num_list:
            top_view_path = "{!s}/{:d}.bin".format(data_dir, frame_num)
            label_path = "{!s}/{:d}.png".format(label_dir, frame_num)
            label_info = misc.imread(label_path)
            top_view_data = self.readstruct_4(top_view_path)
            label_info=label_info.astype(np.float32)
            label_batch.append(label_info)
            data_batch.append(top_view_data)

        label_batch = np.concatenate(label_batch, axis=0).reshape(-1, 400, 300, 1)
        data_batch = np.concatenate(data_batch, axis=0).reshape(-1, 400, 300, 3)

        return data_batch,label_batch

    def batch_data_bin_5(self, data_dir, label_dir):  #y2do area는 숫자가 연속적이지않음.
        filename_list = os.listdir(label_dir)  # numb of files
        frame_num_list=self.get_frame_number(filename_list)
        np.random.shuffle(frame_num_list)
        label_batch, data_batch = [], []
        for frame_num in frame_num_list:
            print("frame number ", frame_num)
            top_view_path = "{!s}/{:d}.bin".format(data_dir, frame_num)
            label_path = "{!s}/{:d}.png".format(label_dir, frame_num)
            label_info = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
            top_view_data = self.readstruct_4(top_view_path)
            label_info=self.change_label_format(label_info)
            label_batch.append(label_info)
            data_batch.append(top_view_data)

        label_batch = np.concatenate(label_batch, axis=0).reshape(-1, 400, 300, 2)
        data_batch = np.concatenate(data_batch, axis=0).reshape(-1, 400, 300, 3)

        return data_batch,label_batch


    def batch_data_bin_6(self, data_dir, label_dir):  #y2do area 는숫자가연소적이지않음.
        filename_list = os.listdir(label_dir)  # numb of files
        frame_num_list=self.get_frame_number(filename_list)
        np.random.shuffle(frame_num_list)
        label_batch, data_batch = [], []
        for frame_num in frame_num_list:
            print("frame_number : ", frame_num)
            top_view_path = "{!s}/{:d}.bin".format(data_dir, frame_num)
            label_path = "{!s}/{:d}.png".format(label_dir, frame_num)
            label_info = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
            top_view_data = self.readstruct_5(top_view_path)
            label_batch.append(label_info)
            data_batch.append(top_view_data)

        label_batch = np.concatenate(label_batch, axis=0).reshape(-1, 400, 300, 1)
        data_batch = np.concatenate(data_batch, axis=0).reshape(-1, 400, 300, 5)

        return data_batch,label_batch





    def change_label_format(self,image):
        image=np.expand_dims(image,axis=2)
        road_color=np.array([1])
        road=np.all(image==road_color,axis=2)
        not_road=np.all(image!=road_color,axis=2)
        label_all=np.dstack([not_road,road])
        label_all=label_all.astype(np.float32)

        return label_all


    def get_frame_number_for_test(self,label_dir):
        filename_list = os.listdir(label_dir)  # numb of files
        frame_num_list=self.get_frame_number(filename_list)

        return frame_num_list


    def get_frame_number(self,filename_list):
        filenum_list=[]
        for file_name in filename_list:
            frame_num=int(file_name[:-4])
            filenum_list.append(frame_num)

        return filenum_list

    def make_path(self,number_set,steps,batch_size=5):
        batch_set=number_set[(steps-1)*batch_size:steps]
        for idx in batch_set:
            data_path=cfg.TRAIN_DATA_PATH_BIN+"/"+str(idx)+".bin"
            label_path=cfg.TRAIN_DATA_LABEL_PATH+"/"+str(idx)+".png"
            print(data_path)

    def loading_data(self,loaded_data_idx_set):
        label_batch, data_batch = [], []
        for frame_num in loaded_data_idx_set:
            top_view_path = "{!s}/{:d}.bin".format(cfg.TRAIN_DATA_PATH_BIN, frame_num)
            label_path = "{!s}/{:d}.png".format(cfg.TRAIN_DATA_LABEL_PATH, frame_num)
            label_info = misc.imread(label_path)
            top_view_data = self.readstruct_4(top_view_path)
            label_batch.append(label_info)
            data_batch.append(top_view_data)

        loaded_label = np.concatenate(label_batch, axis=0).reshape(-1, 400, 300, 1)
        loaded_data = np.concatenate(data_batch, axis=0).reshape(-1, 400, 300, 3)

        return loaded_data,loaded_label







    def temp_batch_data(self ,data_dir,label_dir):# to check shuffle the mini-batch data after every epoch
        filename_list=os.listdir(data_dir)#numb of files
        file_size=len(filename_list)
        rand_idx=np.arange(file_size)
        np.random.shuffle(rand_idx)
        label_batch, data_batch_path=[],[]
        for frame_num in rand_idx:
            top_view_path = "{!s}/{:d}.png".format(data_dir, frame_num)
            data_batch_path.append(top_view_path)

        return data_batch_path


    def readstruct(self):
        x = Struct('iiIffff')#x,y,numofPoints,avh,maxh,minh,varh
        with open('Data/train_data/1000_py.bin', 'rb') as bin_file:
            result = []
            while True:
                buf = bin_file.read(x.size)
                if (len(buf) != x.size):
                    break
                result.append(x.unpack_from(buf))

        return self.list2cvmat(result)

    def readstruct_2(self,path):
        x = Struct('iiIffff')#x,y,numofPoints,avh,maxh,minh,varh
        with open(path, 'rb') as bin_file:
            result = []
            while True:
                buf = bin_file.read(x.size)
                if (len(buf) != x.size):
                    break
                result.append(x.unpack_from(buf))

        return self.list2cvmat_3(result)

    def readstruct_3(self,path):
        x = Struct('iiIffff')#x,y,numofPoints,avh,maxh,minh,varh
        with open(path, 'rb') as bin_file:
            result = []
            while True:
                buf = bin_file.read(x.size)
                if (len(buf) != x.size):
                    break
                result.append(x.unpack_from(buf))

        return self.list2cvmat_4(result)

    def readstruct_4(self,path):
        x = Struct('iiIffff')#x,y,numofPoints,avh,maxh,minh,varh
        with open(path, 'rb') as bin_file:
            result = []
            while True:
                buf = bin_file.read(x.size)
                if (len(buf) != x.size):
                    break
                result.append(x.unpack_from(buf))

        return self.list2cvmat_5(result)


    def readstruct_5(self,path):
        x = Struct('iiIffff')#x,y,numofPoints,avh,maxh,minh,varh
        with open(path, 'rb') as bin_file:
            result = []
            while True:
                buf = bin_file.read(x.size)
                if (len(buf) != x.size):
                    break
                result.append(x.unpack_from(buf))

        return self.list2cvmat_6(result)#5개의 feature 값 모두 사용한 경우.


    def readstruct_6(self,path):
        x = Struct('iiIffff')#x,y,numofPoints,avh,maxh,minh,varh
        with open(path, 'rb') as bin_file:
            result = []
            while True:
                buf = bin_file.read(x.size)
                if (len(buf) != x.size):
                    break
                result.append(x.unpack_from(buf))

        return self.list2cvmat_7(result)#5개의 feature 값 모두 사용한 경우.

    def readstruct_7(self,path):#generalize
        x = Struct('iiIffff')
        with open(path, 'rb') as bin_file:
            result = []
            while True:
                buf = bin_file.read(x.size)
                if (len(buf) != x.size):
                    break
                result.append(x.unpack_from(buf))

        return self.list2cvmat_8(result)


    def readstruct_for_rgb_image(self,path):
        x = Struct('iiIffff')#x,y,numofPoints,avh,maxh,minh,variance
        #위 값 중에서 x,y,numbofpoint,s
        with open(path, 'rb') as bin_file:
            result = []
            while True:
                buf = bin_file.read(x.size)
                if (len(buf) != x.size):
                    break
                result.append(x.unpack_from(buf))

        return self.list2cvmat_for_raw_image(result)


    def list2cvmat(self, inputList):
        img = np.zeros((400, 250), np.uint8)
        lateral_bias, vertical_bias = 100, 50

        for i in range(len(inputList)):
            vertical_pos = inputList[i][0] + vertical_bias
            lateral_pos = inputList[i][1] + lateral_bias
            if ((vertical_pos >= 0) and (vertical_pos < 400) and (lateral_pos >= 0) and (lateral_pos < 250)):
                img[vertical_pos, lateral_pos] = 255#여기서 평균높이값
                #print(inputList[i][3])

        flipped_img = cv2.flip(img, -1)
        return flipped_img


    def list2cvmat_2(self, inputList):
        img = np.zeros((400, 250), np.float32)
        lateral_bias, vertical_bias = 100, 50

        for i in range(len(inputList)):
            vertical_pos = inputList[i][0] + vertical_bias
            lateral_pos = inputList[i][1] + lateral_bias
            if ((vertical_pos >= 0) and (vertical_pos < 400) and (lateral_pos >= 0) and (lateral_pos < 250)):
                img[vertical_pos, lateral_pos] = inputList[i][3]#여기서 평균높이값
                #print(inputList[i][3])

        flipped_img = cv2.flip(img, -1)
        return flipped_img


    def list2cvmat_3(self, inputList):
        img = np.zeros((400, 250,3), np.float32)
        lateral_bias, vertical_bias = 100, 50

        for i in range(len(inputList)):
            vertical_pos = inputList[i][0] + vertical_bias
            lateral_pos = inputList[i][1] + lateral_bias
            if ((vertical_pos >= 0) and (vertical_pos < 400) and (lateral_pos >= 0) and (lateral_pos < 250)):
                img[vertical_pos, lateral_pos,0] = inputList[i][3]#여기서 평균높이값
                img[vertical_pos, lateral_pos, 1] = inputList[i][4]
                img[vertical_pos, lateral_pos, 2] = inputList[i][6]
        flipped_img = cv2.flip(img, -1)
        return flipped_img

    def list2cvmat_4(self, inputList):
        img = np.zeros((400, 300,3), np.float32)

        for i in range(len(inputList)):
            vertical_pos = inputList[i][0]
            lateral_pos = inputList[i][1]
            if ((vertical_pos >= 0) and (vertical_pos < 400) and (lateral_pos >= 0) and (lateral_pos < 300)):
                img[vertical_pos, lateral_pos, 0] = inputList[i][2]#numofpoints
                img[vertical_pos, lateral_pos, 1] = inputList[i][4]#max_height
                img[vertical_pos, lateral_pos, 2] = inputList[i][6]#variance
        flipped_img = cv2.flip(img, -1)
        return flipped_img

    def list2cvmat_5(self, inputList):
        img = np.zeros((400, 300,3), np.float32)

        for i in range(len(inputList)):
            vertical_pos = inputList[i][0]
            lateral_pos = inputList[i][1]
            if ((vertical_pos >= 0) and (vertical_pos < 400) and (lateral_pos >= 0) and (lateral_pos < 300)):
                img[vertical_pos, lateral_pos, 0] = inputList[i][2]#numofpoints
                img[vertical_pos, lateral_pos, 1] = inputList[i][4]#max_height
                img[vertical_pos, lateral_pos, 2] = inputList[i][6]#variance
        return img


    def list2cvmat_6(self, inputList): # adding channel
        img = np.zeros((400, 300,5), np.float32)

        for i in range(len(inputList)):
            vertical_pos = inputList[i][0]
            lateral_pos = inputList[i][1]
            if ((vertical_pos >= 0) and (vertical_pos < 400) and (lateral_pos >= 0) and (lateral_pos < 300)):
                img[vertical_pos, lateral_pos, 0] = inputList[i][2]#numofpoints
                img[vertical_pos, lateral_pos, 1] = inputList[i][3]#average_height
                img[vertical_pos, lateral_pos, 2] = inputList[i][4]#max_h
                img[vertical_pos, lateral_pos, 3] = inputList[i][5]#min h
                img[vertical_pos, lateral_pos, 4] = inputList[i][6]#variance height

        return img

    #x, y, numofPoints, avh, maxh, minh, varh


    def list2cvmat_7(self, inputList): # adding channel
        img = np.zeros((640, 480,3), np.float32)
        for i in range(len(inputList)):
            vertical_pos = inputList[i][0]
            lateral_pos = inputList[i][1]
            if ((vertical_pos >= 0) and (vertical_pos < 640) and (lateral_pos >= 0) and (lateral_pos < 480)):
                img[vertical_pos, lateral_pos, 0] = inputList[i][2]#numofpoints
                img[vertical_pos, lateral_pos, 1] = inputList[i][4]#max_height
                img[vertical_pos, lateral_pos, 2] = inputList[i][6]#variance

        return img

    #x, y, numofPoints, avh, maxh, minh, varh


    def list2cvmat_8(self, inputList):
        img = np.zeros((cfg.IMAGE_H, cfg.IMAGE_W,3), np.float32)

        for i in range(len(inputList)):
            vertical_pos = inputList[i][0]
            lateral_pos = inputList[i][1]
            if ((vertical_pos >= 0) and (vertical_pos < cfg.IMAGE_H) and (lateral_pos >= 0) and (lateral_pos < cfg.IMAGE_W)):
                img[vertical_pos, lateral_pos, 0] = inputList[i][2]#numofpoints
                img[vertical_pos, lateral_pos, 1] = inputList[i][4]#max_height
                img[vertical_pos, lateral_pos, 2] = inputList[i][6]#variance
        return img




    def list2cvmat_for_raw_image(self, inputList):
        img = np.zeros((400, 250,3), np.float32)
        lateral_bias, vertical_bias = 100, 50

        for i in range(len(inputList)):
            vertical_pos = inputList[i][0] + vertical_bias
            lateral_pos = inputList[i][1] + lateral_bias
            if ((vertical_pos >= 0) and (vertical_pos < 400) and (lateral_pos >= 0) and (lateral_pos < 250)):
                img[vertical_pos, lateral_pos,0] = 255
                img[vertical_pos, lateral_pos, 1] = 255
                img[vertical_pos, lateral_pos, 2] = 255
        flipped_img = cv2.flip(img, -1)
        return flipped_img


    def batch_read_input_file(self,image_filenames,step,batch_size):
        x = Struct('iiIffff')  # x,y,numofPoints,avh,maxh,minh,varh
        batch_image = np.empty([batch_size, 400, 250, 3], dtype=np.float32)
        for i in range(batch_size):

            with open(image_filenames[i], 'rb') as bin_file:
                result = []
                while True:
                    buf = bin_file.read(x.size)
                    if (len(buf) != x.size):
                        break
                    result.append(x.unpack_from(buf))
            image = self.list2cvmat_3(result)
            batch_image[i, :, :, :] = image


        return batch_image

    def getGroundTruth(self,fileNameGT):
        '''
        Returns the ground truth maps for roadArea and the validArea
        :param fileNameGT:
        '''
        # Read GT
        assert os.path.isfile(fileNameGT), 'Cannot find: %s' % fileNameGT
        full_gt = cv2.imread(fileNameGT, cv2.IMREAD_GRAYSCALE)
        # attention: OpenCV reads in as BGR, so first channel has Blue / road GT
        roadArea = full_gt[:, :] > 0
        validArea = full_gt[:, :] > 0

        return roadArea, validArea

    def getGroundTruth_label_image(self,full_gt):
        '''
        Returns the ground truth maps for roadArea and the validArea
        :param fileNameGT:
        '''
        # Read GT
        # attention: OpenCV reads in as BGR, so first channel has Blue / road GT
        roadArea = full_gt[:, :] > 0
        validArea = full_gt[:, :] > 0

        return roadArea, validArea

    def getEvaluation(self, gtBin, cur_prob):
        thres = np.array(range(0, 256)) / 255
        ref_idx=np.argmax(thres>=0.5)
        cur_prob=np.clip((cur_prob.astype('f4'))/np.iinfo(cur_prob.dtype).max,0,1)#normalize 0~1


        thresInf=np.concatenate(([-np.Inf], thres, [np.Inf]))
        fnArray = cur_prob[(gtBin == True)]#get the probability of road area.
        fnHist = np.histogram(fnArray, bins=thresInf)[0]#[0] is the number of elements in each interval, second is the interval
        fnCum = np.cumsum(fnHist)  # 사실 이게threshold 값 기준으로 짜르기  쉽도록 histogram 으로나타내었다. thresInf size if 258(256+2), the number of interval is 257. so fnHist is also 257
        FN = fnCum[ref_idx]  # len(thres) is 256, fnCum size is 257 maybe last interval is 1~np.inf so, exclude this interval.
        #here we chnage the len(thres) to another value, we get the number of false negative.

        fpArray = cur_prob[(gtBin == False)]# 도로가 아닌 부분의pixel 의 확률값 만 뽑아서
        print(fpArray)
        print(len(fpArray))
        fpHist = np.histogram(fpArray, bins=thresInf)[0]  # histogram 을뽑는다.  이거는 반대로 확류값이 낮게 나오는게 더 좋으니깐
        fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))  # acc 를inverse시킨다. 즉,   왼쪽에서 오른쪽으로 확률이 높은것에서 낮은 쪽으로
        FP = fpCum[255-ref_idx]  # 위에는0  부터 하고 여기는 왜1 부터 하는지는  좀헷갈린다. because 0 is the interval between 1~ np.inf. so no value exists in that eixts.

        posNum=np.sum(gtBin==True)#num of pixels within the road area
        negNum=np.sum(gtBin==False)

        TP = posNum - FN #num of road area -
        TN = negNum - FP #num of non

        # valid=(TP>=0)&(TN>=0)
        # assert valid.all(), "Detected invalid elements in eval"



        recall = TP / float(posNum)
        precision = TP / (TP + FP + 1e-10)
        print(recall)
        print(precision)

        return recall, precision,fnArray,fpArray

    def getEvaluation_2(self,gtBin,cur_prob,cut_value):
        thres = np.array(range(0, 256)) / 255
        ref_idx = np.argmax(thres >= cut_value)
        cur_prob = np.clip((cur_prob.astype('f4')) / np.iinfo(cur_prob.dtype).max, 0, 1)  # normalize 0~1

        thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
        fnArray = cur_prob[(gtBin == True)]  # get the probability of road area.
        fnHist = np.histogram(fnArray, bins=thresInf)[0]  # [0] is the number of elements in each interval, second is the interval
        fnCum = np.cumsum(fnHist)  # 사실 이게threshold 값 기준으로 짜르기  쉽도록 histogram 으로나타내었다. thresInf size if 258(256+2), the number of interval is 257. so fnHist is also 257
        FN = fnCum[ref_idx]  # len(thres) is 256, fnCum size is 257 maybe last interval is 1~np.inf so, exclude this interval.
        # here we chnage the len(thres) to another value, we get the number of false negative.

        fpArray = cur_prob[(gtBin == False)]  # 도로가 아닌 부분의pixel 의 확률값 만 뽑아서
        fpHist = np.histogram(fpArray, bins=thresInf)[0]  # histogram 을뽑는다.  이거는 반대로 확류값이 낮게 나오는게 더 좋으니깐
        fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))  # acc 를inverse시킨다. 즉,   왼쪽에서 오른쪽으로 확률이 높은것에서 낮은 쪽으로
        FP = fpCum[255 - ref_idx]  # 위에는0  부터 하고 여기는 왜1 부터 하는지는  좀헷갈린다. because 0 is the interval between 1~ np.inf. so no value exists in that eixts.

        posNum = np.sum(gtBin == True)  # num of pixels within the road area
        negNum = np.sum(gtBin == False)

        TP = posNum - FN  # num of road area -
        TN = negNum - FP  # num of non

        # valid=(TP>=0)&(TN>=0)
        # assert valid.all(), "Detected invalid elements in eval"

        recall = TP / float(posNum)
        precision = TP / (TP + FP + 1e-10)
        print(recall)
        print(precision)

        return recall, precision, fnArray, fpArray


    def getEvaluation_3(self,gtBin,cur_prob):#following kitti evaluation code
        thres = np.array(range(0, 256)) / 255
        cur_prob = np.clip((cur_prob.astype('f4')) / np.iinfo(cur_prob.dtype).max, 0, 1)  # normalize 0~1

        thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
        fnArray = cur_prob[(gtBin == True)]  # get the probability of road area.
        fnHist = np.histogram(fnArray, bins=thresInf)[0]  # [0] is the number of elements in each interval, second is the interval
        fnCum = np.cumsum(fnHist)  # 사실 이게threshold 값 기준으로 짜르기  쉽도록 histogram 으로나타내었다. thresInf size if 258(256+2), the number of interval is 257. so fnHist is also 257
        FN = fnCum[0:0+len(thres)]  # len(thres) is 256, fnCum size is 257 maybe last interval is 1~np.inf so, exclude this interval.
        # here we chnage the len(thres) to another value, we get the number of false negative.

        fpArray = cur_prob[(gtBin == False)]  # 도로가 아닌 부분의pixel 의 확률값 만 뽑아서
        fpHist = np.histogram(fpArray, bins=thresInf)[0]  # histogram 을뽑는다.  이거는 반대로 확류값이 낮게 나오는게 더 좋으니깐
        fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))  # acc 를inverse시킨다. 즉,   왼쪽에서 오른쪽으로 확률이 높은것에서 낮은 쪽으로
        FP = fpCum[1:1+len(thres)]  # 위에는0  부터 하고 여기는 왜1 부터 하는지는  좀헷갈린다. because 0 is the interval between 1~ np.inf. so no value exists in that eixts.

        posNum = np.sum(gtBin == True)  # num of pixels within the road area
        negNum = np.sum(gtBin == False)

        TP = posNum - FN  # num of road area -
        TN = negNum - FP  # num of non


        valid=(TP>=0)&(TN>=0)
        assert valid.all(), "Detected invalid elements in eval" #모든element가true여야한다.

        Final_result=self.pxEval_maximizeFMeasure(posNum,negNum,FN,FP,thresh=thres)


        return Final_result


    def pxEval_maximizeFMeasure(self, totalPosNum, totalNegNum, totalFN, totalFP, thresh=None):
        '''

        @param totalPosNum: scalar
        @param totalNegNum: scalar
        @param totalFN: vector
        @param totalFP: vector
        @param thresh: vector
        '''

        # Calc missing stuff
        totalTP = totalPosNum - totalFN  # 여기서 구하고자 하는  것은  알고리즘이  구한 true positive이다. 즉,
        # 전체 road  부분에 해당하는 픽셀 수 에서  도로가 아니라고 판단한pixel 수의 개수를 빼면  도로영역에서 도로라고 판단한pixel 의 개수가 나오는것이아닌가?
        totalTN = totalNegNum - totalFP  # 비슷하게 도로가 아닌 영역에서  도로라고 판단한pixel 의 수를 빼면  도로가 아니 영역에서 도로가 아니라고  판단한pixel 의 개수가 나오게된다.

        valid = (totalTP >= 0) & (totalTN >= 0)
        assert valid.all(), 'Detected invalid elements in eval'

        recall = totalTP / float(totalPosNum)
        precision = totalTP / (totalTP + totalFP + 1e-10)

        selector_invalid = (recall == 0) & (precision == 0)
        recall = recall[~selector_invalid]
        precision = precision[~selector_invalid]

        maxValidIndex = len(precision)

        # Pascal VOC average precision
        AvgPrec = 0
        counter = 0
        for i in np.arange(0, 1.1, 0.1):
            ind = np.where(recall >= i)
            if ind == None:
                continue
            pmax = max(precision[ind])
            AvgPrec += pmax
            counter += 1
        AvgPrec = AvgPrec / counter

        # F-measure operation point
        beta = 1.0
        betasq = beta ** 2
        F = (1 + betasq) * (precision * recall) / ((betasq * precision) + recall + 1e-10)
        index = F.argmax()
        MaxF = F[index]

        recall_bst = recall[index]
        precision_bst = precision[index]

        TP = totalTP[index]
        TN = totalTN[index]
        FP = totalFP[index]
        FN = totalFN[index]
        valuesMaxF = np.zeros((1, 4), 'u4')
        valuesMaxF[0, 0] = TP
        valuesMaxF[0, 1] = TN
        valuesMaxF[0, 2] = FP
        valuesMaxF[0, 3] = FN

        # ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)
        prob_eval_scores = self.calcEvalMeasures(valuesMaxF)
        prob_eval_scores['AvgPrec'] = AvgPrec
        prob_eval_scores['MaxF'] = MaxF

        # prob_eval_scores['totalFN'] = totalFN
        # prob_eval_scores['totalFP'] = totalFP
        prob_eval_scores['totalPosNum'] = totalPosNum
        prob_eval_scores['totalNegNum'] = totalNegNum

        prob_eval_scores['precision'] = precision
        prob_eval_scores['recall'] = recall
        prob_eval_scores['precision_bst'] = precision_bst
        prob_eval_scores['recall_bst'] = recall_bst
        prob_eval_scores['thresh'] = thresh
        print("AvgPrec=", AvgPrec," MaxF=",MaxF," Precision_bst=",precision_bst," Recall_bst=",recall_bst)
        # if thresh != None:
        if thresh is not None:
            BestThresh = thresh[index]
            prob_eval_scores['BestThresh'] = BestThresh

        # return a dict
        return prob_eval_scores

    def calcEvalMeasures(self, evalDict, tag='_wp'):
        '''

        :param evalDict:
        :param tag:
        '''
        # array mode!
        TP = evalDict[:, 0].astype('f4')
        TN = evalDict[:, 1].astype('f4')
        FP = evalDict[:, 2].astype('f4')
        FN = evalDict[:, 3].astype('f4')
        Q = TP / (TP + FP + FN)
        P = TP + FN
        N = TN + FP
        TPR = TP / P
        FPR = FP / N
        FNR = FN / P
        TNR = TN / N
        A = (TP + TN) / (P + N)
        precision = TP / (TP + FP)
        recall = TP / P
        # numSamples = TP + TN + FP + FN
        correct_rate = A

        # F-measure
        # beta = 1.0
        # betasq = beta**2
        # F_max = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)

        outDict = dict()

        outDict['TP' + tag] = TP
        outDict['FP' + tag] = FP
        outDict['FN' + tag] = FN
        outDict['TN' + tag] = TN
        outDict['Q' + tag] = Q
        outDict['A' + tag] = A
        outDict['TPR' + tag] = TPR
        outDict['FPR' + tag] = FPR
        outDict['FNR' + tag] = FNR
        outDict['PRE' + tag] = precision
        outDict['REC' + tag] = recall
        outDict['correct_rate' + tag] = correct_rate
        return outDict


    def spatial_dropout(self,x, keep_prob):
        # x is a convnet activation with shape BxWxHxF where F is the
        # number of feature maps for that layer
        # keep_prob is the proportion of feature maps we want to keep

        # get the batch size and number of feature maps
        num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]

        # get some uniform noise between keep_prob and 1 + keep_prob
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(num_feature_maps,dtype=x.dtype)

        # if we take the floor of this, we get a binary matrix where
        # (1-keep_prob)% of the values are 0 and the rest are 1
        binary_tensor = tf.floor(random_tensor)

        # Reshape to multiply our feature maps by this tensor correctly
        binary_tensor = tf.reshape(binary_tensor,
                                   [-1, 1, 1, tf.shape(x)[3]])

        # Zero out feature maps where appropriate; scale up to compensate
        ret = tf.div(x, keep_prob) * binary_tensor
        return ret, binary_tensor


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

    def shuffle_batch_data_2(self,step,number_set,train_data,label_data,batch_size=5):#change for label format
        batch_number=number_set[(step-1)*batch_size:step*batch_size]
        batch_data=[]
        batch_label=[]
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_label.append(label_data[idx])


        loaded_label = np.concatenate(batch_label, axis=0).reshape(-1, 400, 300, 2)
        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, 400, 300, 3)

        return loaded_data, loaded_label

    def shuffle_batch_data_3(self, step, number_set, train_data, fsd_label,obj_label, batch_size=5):  # change for label format
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        batch_data, batch_fsd,batch_obj = [],[],[]
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_fsd.append(fsd_label[idx])
            batch_obj.append(obj_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, 400, 300, 3)
        loaded_fsd = np.concatenate(batch_fsd, axis=0).reshape(-1, 400, 300, 1)
        loaded_obj=np.concatenate(batch_obj, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS,6)

        return loaded_data, loaded_fsd,loaded_obj

    def shuffle_batch_data_4(self, step, number_set, train_data, fsd_label, batch_size=5):  #for input data channel( 3-> 5)
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        batch_data, batch_fsd = [],[]
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_fsd.append(fsd_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, 400, 300, 5)
        loaded_fsd = np.concatenate(batch_fsd, axis=0).reshape(-1, 400, 300, 1)

        return loaded_data, loaded_fsd

    def shuffle_batch_data_5(self, step, number_set, train_data, fsd_label, obj_label,batch_size=5):  # change for label format
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        batch_data, batch_fsd, batch_obj = [], [], []
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_fsd.append(fsd_label[idx])
            batch_obj.append(obj_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 3)
        loaded_fsd = np.concatenate(batch_fsd, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 1)
        loaded_obj = np.concatenate(batch_obj, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)

        return loaded_data, loaded_fsd, loaded_obj


    def shuffle_batch_data_6(self, step, number_set, train_data, fsd_label, obj_label,batch_size=5):  # for 400x300
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        batch_data, batch_fsd, batch_obj = [], [], []
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_fsd.append(fsd_label[idx])
            batch_obj.append(obj_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 3)
        loaded_fsd = np.concatenate(batch_fsd, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 1)
        loaded_obj = np.concatenate(batch_obj, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)

        return loaded_data, loaded_fsd, loaded_obj


    def shuffle_batch_data_7(self, step, number_set, train_data, fsd_label, obj_label,frame_num_list,batch_size=5):  # for 400x300
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        first_frame_number=frame_num_list[batch_number[0]]
        batch_data, batch_fsd, batch_obj = [], [], []
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_fsd.append(fsd_label[idx])
            batch_obj.append(obj_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 7)
        loaded_fsd = np.concatenate(batch_fsd, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 1)
        loaded_obj = np.concatenate(batch_obj, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)

        return loaded_data, loaded_fsd, loaded_obj,first_frame_number

    def shuffle_batch_data_bd(self, step, number_set, train_data, fsd_label, frame_num_list,batch_size=5):
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        first_frame_number=frame_num_list[batch_number[0]]
        batch_data, batch_fsd= [], []
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_fsd.append(fsd_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 6)
        loaded_fsd = np.concatenate(batch_fsd, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 1)

        return loaded_data, loaded_fsd, first_frame_number


    def shuffle_batch_data_bd_2(self, step, number_set, train_data, fsd_label, obj_label, frame_num_list,batch_size=5):#adding object label
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        first_frame_number=frame_num_list[batch_number[0]]
        batch_data, batch_fsd, batch_obj = [], [], []
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_fsd.append(fsd_label[idx])
            batch_obj.append(obj_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 7)
        loaded_fsd = np.concatenate(batch_fsd, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 1)
        loaded_obj = np.concatenate(batch_obj, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)

        return loaded_data, loaded_fsd,loaded_obj, first_frame_number




    def shuffle_batch_data_8(self, step, number_set, train_data,obj_label,frame_num_list,batch_size=5):  # for 400x300
        batch_number = number_set[(step - 1) * batch_size:step * batch_size]
        first_frame_number=frame_num_list[batch_number[0]]
        batch_data, batch_obj = [], []
        for idx in batch_number:
            batch_data.append(train_data[idx])
            batch_obj.append(obj_label[idx])

        loaded_data = np.concatenate(batch_data, axis=0).reshape(-1, cfg.IMAGE_H, cfg.IMAGE_W, 7)
        loaded_obj = np.concatenate(batch_obj, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)

        return loaded_data, loaded_obj,first_frame_number




    def read_anchors_file(self,file_path):

        anchors = []
        with open(file_path, 'r') as file:
            for line in file.read().splitlines():
                temp_result = line.split()
                anchors.append(list(map(float, line.split())))

        return np.array(anchors)

    def iou_wh(self,r1, r2):
        # print(r1)
        # print(r2)
        min_w = min(r1[0], r2[0])
        min_h = min(r1[1], r2[1])
        area_r1 = r1[0] * r1[1]  # 각각의 box 넓이
        area_r2 = r2[0] * r2[1]  # 각가의 box 넓

        intersect = min_w * min_h
        union = area_r1 + area_r2 - intersect

        return intersect / union

    def get_active_anchors(self, roi, anchors):  # roi는 gt를 말하고 anchors는 사전에 gt를 조사해서 많이 나오는 box 크기라고 생각하면 된다.

        indxs = []  # rois ? ,? ,width, height
        iou_max, index_max = 0, 0
        for i, a in enumerate(anchors):  # number와 element를 동시에 주는 것이다.
            iou = self.iou_wh(roi[2:], a)  # 여기서는 위치는 상관없이 겨치는 것만 보기 때문에 roi에서 2번 이상인 값부터 넣는다고 보면된다.
            if iou > cfg.TRAINING_IOU_TH:  # 특정 gt가 anchor와 겹치는게 많으면
                indxs.append(i)  # 여기서 indx를 추가.
            if iou > iou_max:
                iou_max, index_max = iou, i  # 여기서는 가장 큰 iou값과 해당 anchor의 idx값을 찾는다.

        if len(indxs) == 0:
            indxs.append(index_max)

        return indxs


    def get_active_anchors_2(self, roi, anchors):  # roi는 gt를 말하고 anchors는 사전에 gt를 조사해서 많이 나오는 box 크기라고 생각하면 된다.

        indxs = []  # rois ? ,? ,width, height
        iou_max, index_max = 0, 0
        for i, a in enumerate(anchors):  # number와 element를 동시에 주는 것이다.
            iou = self.iou_wh(roi[2:4], a)  # 여기서는 위치는 상관없이 겨치는 것만 보기 때문에 roi에서 2번 이상인 값부터 넣는다고 보면된다.
            if iou > cfg.TRAINING_IOU_TH:  # 특정 gt가 anchor와 겹치는게 많으면
                indxs.append(i)  # 여기서 indx를 추가.
            if iou > iou_max:
                iou_max, index_max = iou, i  # 여기서는 가장 큰 iou값과 해당 anchor의 idx값을 찾는다.

        if len(indxs) == 0:
            indxs.append(index_max)

        return indxs



    def get_grid_cell(self,roi, raw_w, raw_h, grid_w, grid_h):

        x_center = roi[0] + roi[2] / 2.0
        y_center = roi[1] + roi[3] / 2.0

        grid_x = int(x_center / float(raw_w) * float(grid_w))
        grid_y = int(y_center / float(raw_h) * float(grid_h))

        return grid_x, grid_y

    def get_grid_cell_2(self,roi, raw_w, raw_h, grid_w, grid_h):
        #, 400, 15, 20
        x_center = roi[0]
        y_center = roi[1]

        grid_x = int(x_center / float(raw_w) * float(grid_w))
        grid_y = int(y_center / float(raw_h) * float(grid_h))

        return grid_x, grid_y #몇 번째 grid에 속하는 가.



    def roi2label(self,roi, anchor, raw_w, raw_h, grid_w, grid_h):  # 여기서 roi가 bounding box인것 같고.

        x_center = roi[0] + roi[2] / 2.0
        y_center = roi[1] + roi[3] / 2.0
        #print("previous : ", x_center,y_center,roi[2],roi[3])
        grid_x = x_center / float(raw_w) * float(grid_w)  # 어떤 grid에 속하는지 나오는데 여기는 float값을 그대로 사용
        grid_y = y_center / float(raw_h) * float(grid_h)  #

        grid_x_offset = grid_x - int(grid_x)  # 해당 grid에서 얼마나 떨어져있는가를 조사.
        grid_y_offset = grid_y - int(grid_y)
        # 여를 들면 하나의 grid는 여러개의 pixel로 이루어져 있다. 즉 1개의 grid가 12x12라고 하면 그 grid 내에서 얼마나 얼마나 떨어져 있는가를 offset이라고 한다.

        roi_w_scale = roi[2] / anchor[0]  # gt와 anchor의 비율을 정하는 것이다. 즉 gt_w/anchor_w
        roi_h_scale = roi[3] / anchor[1]

        label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]

        return label

    def roi2label_2(self,roi, anchor, raw_w, raw_h, grid_w, grid_h):  # 여기서 roi가 bounding box인것 같고.

        x_center = roi[0]
        y_center = roi[1]
        #print("current : ", x_center, y_center,roi[2],roi[3])
        grid_x = x_center / float(raw_w) * float(grid_w)  # 어떤 grid에 속하는지 나오는데 여기는 float값을 그대로 사용
        grid_y = y_center / float(raw_h) * float(grid_h)  #
        #ex) grid_x 4.3 이라고 하면 4번째 grid에서 0.3만큼 떨어진 곳에 있다고 생각.
        grid_x_offset = grid_x - int(grid_x)  # 해당 grid에서 얼마나 떨어져있는가를 조사.
        grid_y_offset = grid_y - int(grid_y)
        # 여를 들면 하나의 grid는 여러개의 pixel로 이루어져 있다. 즉 1개의 grid가 12x12라고 하면 그 grid 내에서 얼마나 얼마나 떨어져 있는가를 offset이라고 한다.

        roi_w_scale = roi[2] / anchor[0]  # gt와 anchor의 비율을 정하는 것이다. 즉 gt_w/anchor_w
        roi_h_scale = roi[3] / anchor[1]

        label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]

        return label




    def test_fsd_yolo_data(self,input_data_ph,fsd_label_ph,yolo_label_ph):
        input_data=self.readstruct_4(input_data_ph)
        label_info=cv2.imread(fsd_label_ph,cv2.IMREAD_GRAYSCALE)
        label_info = self.change_label_format(label_info)
        yolo_label=self.get_yolo_label()


        return input_data,label_info,yolo_label


    def fsd_yolo_data_0(self,fsd_dir_path):#fsd label format이 cross entropy에 맞는것.
        file_list=glob.glob(fsd_dir_path)
        data_list, fsd_label_list, obj_label_list, frame_num_list=[],[],[],[]
        for file_name in file_list:
            frame_number=file_name.split('/')[-1].split('.')[0]
            input_data_path=cfg.ROOT_DIR+'bin_file/'+str(frame_number)+'.bin'
            obj_label_path=cfg.ROOT_DIR+'object_info_2/'+str(frame_number)+'.txt'

            data=self.readstruct_4(input_data_path)
            tmp_fsd_label=cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
            fsd_label = self.change_label_format(tmp_fsd_label)
            obj_label=self.get_yolo_label_1(obj_label_path)

            data_list.append(data)
            fsd_label_list.append(fsd_label)
            obj_label_list.append(obj_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 400, 300, 3)
        fsd_batch = np.concatenate(fsd_label_list, axis=0).reshape(-1, 400, 300, 2)
        obj_batch = np.concatenate(obj_label_list, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 6)

        return data_batch, fsd_batch, obj_batch,frame_num_list



    def fsd_yolo_data_1(self,fsd_dir_path):#fsd label format이 sparse cross entophy에 맺는것.
        file_list=glob.glob(fsd_dir_path)
        data_list, fsd_label_list, obj_label_list, frame_num_list=[],[],[],[]
        for file_name in file_list:
            frame_number=file_name.split('/')[-1].split('.')[0]
            input_data_path=cfg.ROOT_DIR+'bin_file/'+str(frame_number)+'.bin'
            obj_label_path=cfg.ROOT_DIR+'object_info_2/'+str(frame_number)+'.txt'

            data=self.readstruct_4(input_data_path)
            fsd_label=cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
            obj_label=self.get_yolo_label_1(obj_label_path)

            data_list.append(data)
            fsd_label_list.append(fsd_label)
            obj_label_list.append(obj_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 400, 300, 3)
        fsd_batch = np.concatenate(fsd_label_list, axis=0).reshape(-1, 400, 300, 1)
        obj_batch = np.concatenate(obj_label_list, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 6)

        return data_batch, fsd_batch, obj_batch,frame_num_list








    def fsd_yolo_data_2(self,fsd_dir_path):##이것은 heading angle까지 포함한 데이터
        file_list=glob.glob(fsd_dir_path)
        data_list, fsd_label_list, obj_label_list, frame_num_list=[],[],[],[]
        for file_name in file_list:
            print(file_name)
            frame_number = file_name.split('/')[-1].split('.')[0]
            print(frame_number)
            input_data_path = cfg.ROOT_DIR + 'bin_file/' + str(frame_number) + '.bin'
            obj_label_path = cfg.ROOT_DIR + 'object_info/' + str(frame_number) + '.txt'

            data = self.readstruct_4(input_data_path)
            fsd_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            obj_label = self.get_obj_label_4(obj_label_path)

            # total_data_list.append([data,fsd_label,obj_label,frame_number])
            data_list.append(data)
            fsd_label_list.append(fsd_label)
            obj_label_list.append(obj_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 400, 300, 3)
        fsd_batch = np.concatenate(fsd_label_list, axis=0).reshape(-1, 400, 300, 1)
        obj_batch = np.concatenate(obj_label_list, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)
        #여기는 크게 값 자체에 대해서 큰 문제는 없는 듯 하다.

        return data_batch, fsd_batch, obj_batch,frame_num_list

    def fsd_yolo_data_3(self,fsd_dir_path):#heading+new data type
        file_list=glob.glob(fsd_dir_path)
        data_list, fsd_label_list, obj_label_list, frame_num_list=[],[],[],[]
        for file_name in file_list:
            #print(file_name)
            frame_number = file_name.split('/')[-1].split('.')[0]
            #print(frame_number)
            input_data_path = cfg.ROOT_DIR + 'bin_file/' + str(frame_number) + '.bin'
            obj_label_path = cfg.ROOT_DIR + 'object_info_2/' + str(frame_number) + '.txt'

            data = self.readstruct_6(input_data_path)
            fsd_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            obj_label = self.get_obj_label_3(obj_label_path)

            # total_data_list.append([data,fsd_label,obj_label,frame_number])
            data_list.append(data)
            fsd_label_list.append(fsd_label)
            obj_label_list.append(obj_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1,cfg.IMAGE_H , cfg.IMAGE_W, 3)
        fsd_batch = np.concatenate(fsd_label_list, axis=0).reshape(-1, cfg.IMAGE_H , cfg.IMAGE_W, 1)
        obj_batch = np.concatenate(obj_label_list, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)
        #여기는 크게 값 자체에 대해서 큰 문제는 없는 듯 하다.

        return data_batch, fsd_batch, obj_batch,frame_num_list

    def fsd_yolo_data_4(self,fsd_dir_path):##416x320
        file_list=glob.glob(fsd_dir_path)
        data_list, fsd_label_list, obj_label_list, frame_num_list=[],[],[],[]
        for file_name in file_list:
            #print(file_name)
            frame_number = file_name.split('/')[-1].split('.')[0]
            #print(frame_number)
            input_data_path = cfg.ROOT_DIR + 'bin_file/' + str(frame_number) + '.bin'
            obj_label_path = cfg.ROOT_DIR + 'object_info/' + str(frame_number) + '.txt'

            data = self.readstruct_7(input_data_path)
            fsd_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            obj_label = self.get_obj_label_4(obj_label_path)

            # total_data_list.append([data,fsd_label,obj_label,frame_number])
            data_list.append(data)
            fsd_label_list.append(fsd_label)
            obj_label_list.append(obj_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 3)
        fsd_batch = np.concatenate(fsd_label_list, axis=0).reshape(-1, 416, 320, 1)
        obj_batch = np.concatenate(obj_label_list, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)
        #여기는 크게 값 자체에 대해서 큰 문제는 없는 듯 하다.

        return data_batch, fsd_batch, obj_batch,frame_num_list


    def fsd_yolo_data_5(self,fsd_dir_path):##416x320
        file_list=glob.glob(fsd_dir_path)
        data_list, fsd_label_list, obj_label_list, frame_num_list=[],[],[],[]
        for file_name in file_list:
            #print(file_name)
            frame_number = file_name.split('/')[-1].split('.')[0]
            print(frame_number)
            obj_label_path = cfg.ROOT_DIR + 'object_info/' + str(frame_number) + '.txt'

            data = self.get_slice_data(frame_number)
            fsd_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            obj_label = self.get_obj_label_4(obj_label_path)

            # total_data_list.append([data,fsd_label,obj_label,frame_number])
            data_list.append(data)
            fsd_label_list.append(fsd_label)
            obj_label_list.append(obj_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 7)
        fsd_batch = np.concatenate(fsd_label_list, axis=0).reshape(-1, 416, 320, 1)
        obj_batch = np.concatenate(obj_label_list, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)
        #여기는 크게 값 자체에 대해서 큰 문제는 없는 듯 하다.

        return data_batch, fsd_batch, obj_batch,frame_num_list

    def bd_data_gen(self,bd_dir_path):##416x320
        file_list=glob.glob(bd_dir_path)
        data_list, bd_label_list, obj_label_list, frame_num_list=[],[],[],[]
        for file_name in file_list:
            frame_number = file_name.split('/')[-1].split('.')[0]
            obj_label_path = cfg.ROOT_DIR + 'object_info/' + str(frame_number) + '.txt'
            bd_label_path=cfg.ROOT_DIR+'boundary_image/binary_image/'+ str(frame_number)+'.png'
            data = self.get_slice_data(frame_number)
            bd_label = cv2.imread(bd_label_path, cv2.IMREAD_GRAYSCALE)
            obj_label = self.get_obj_label_4(obj_label_path)

            data_list.append(data)
            bd_label_list.append(bd_label)
            obj_label_list.append(obj_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 7)
        bd_batch = np.concatenate(bd_label_list, axis=0).reshape(-1, 416, 320, 1)
        obj_batch = np.concatenate(obj_label_list, axis=0).reshape(-1, cfg.GRID_H, cfg.GRID_W, cfg.N_ANCHORS, 8)
        #여기는 크게 값 자체에 대해서 큰 문제는 없는 듯 하다.

        return data_batch, bd_batch, obj_batch,frame_num_list



    def bd_data_gen_2(self,bd_dir_path):##416x320
        file_list=glob.glob(bd_dir_path)
        data_list, bd_label_list, frame_num_list=[],[],[]
        for file_name in file_list:
            #print(file_name)
            frame_number = file_name.split('/')[-1].split('.')[0]
            bd_label_path=cfg.ROOT_DIR+'boundary_image/binary_image/'+ str(frame_number)+'.png'
            data = self.get_slice_data(frame_number)
            bd_label = cv2.imread(bd_label_path, cv2.IMREAD_GRAYSCALE)

            data_list.append(data)
            bd_label_list.append(bd_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 7)
        bd_batch = np.concatenate(bd_label_list, axis=0).reshape(-1, 416, 320, 1)

        return data_batch, bd_batch, frame_num_list

    def bd_data_gen_3(self,bd_dir_path,root_path):##416x320
        file_list=glob.glob(bd_dir_path)
        data_list, bd_label_list, frame_num_list=[],[],[]
        for file_name in file_list:
            frame_number = file_name.split('/')[-1].split('.')[0]
            bd_label_path=root_path+'boundary_image/binary/'+ str(frame_number)+'.png'
            data = self.get_slice_data_2(frame_number,root_path)
            bd_label = cv2.imread(bd_label_path, cv2.IMREAD_GRAYSCALE)

            data_list.append(data)
            bd_label_list.append(bd_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 6)
        bd_batch = np.concatenate(bd_label_list, axis=0).reshape(-1, 416, 320, 1)

        return data_batch, bd_batch, frame_num_list

    def bd_data_gen_4(self,bd_dir_path,root_path):##416x320 for total_dataset
        file_list=glob.glob(bd_dir_path)
        data_list, bd_label_list, frame_num_list=[],[],[]
        for file_name in file_list:
            frame_number = file_name.split('/')[-1].split('.')[0]
            bd_label_path=root_path+'boundary_image/'+ str(frame_number)+'.png'
            data = self.get_slice_data_2(frame_number,root_path)
            bd_label = cv2.imread(bd_label_path, cv2.IMREAD_GRAYSCALE)

            data_list.append(data)
            bd_label_list.append(bd_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 6)
        bd_batch = np.concatenate(bd_label_list, axis=0).reshape(-1, 416, 320, 1)

        return data_batch, bd_batch, frame_num_list



    def bd_data_gen_5(self,data_path):##general
        file_list=glob.glob(data_path+"boundary_image/*.png")
        data_list, bd_label_list, frame_num_list=[],[],[]
        for file_name in file_list:
            frame_number = file_name.split('/')[-1].split('.')[0]
            data = self.get_slice_data_2(frame_number,data_path)#density map, slice height map(5)
            bd_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            data_list.append(data)
            bd_label_list.append(bd_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 6)
        bd_batch = np.concatenate(bd_label_list, axis=0).reshape(-1, 416, 320, 1)

        return data_batch, bd_batch, frame_num_list


    def bd_data_gen_6(self,data_path):#201908018, height slice에서 0번 빼고 visual cue를 대신 직업 넣음
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



    def test_data_gen(self,data_path,root_path):##general
        file_list=glob.glob(data_path+"boundary_image/*.png")
        data_list, bd_label_list, frame_num_list=[],[],[]
        for file_name in file_list:
            frame_number = file_name.split('/')[-1].split('.')[0]
            data = self.get_test_slice_data(frame_number,root_path)
            bd_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            data_list.append(data)
            bd_label_list.append(bd_label)
            frame_num_list.append(frame_number)

        data_batch = np.concatenate(data_list, axis=0).reshape(-1, 416, 320, 6)
        bd_batch = np.concatenate(bd_label_list, axis=0).reshape(-1, 416, 320, 1)

        return data_batch, bd_batch, frame_num_list
    
    
    def test_data_gen_2(self,data_path,root_path):#20190822this function exploits the visual cue instread of using height map idx 0
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
    
    
    

    def get_test_slice_data(self, frame_number,root_path):
        density_path=root_path+"input_data/density_map/"+str(frame_number)+".png"
        density_image=cv2.imread(density_path,cv2.IMREAD_GRAYSCALE)
        current_slice_set=[]
        for i in range(5):
            slice_path=root_path+"input_data/slice_map/"+str(frame_number)+"_"+str(i)+".png"
            slice_image=cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            current_slice_set.append(slice_image)

        current_slice_set.append(density_image)
        total_data_set=np.concatenate(current_slice_set,axis=0).reshape(416,320,6)
        return total_data_set

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


    def get_slice_data(self, frame_number):
        density_path=cfg.ROOT_DIR+"input_data/density_map/"+str(frame_number)+".png"
        density_image=cv2.imread(density_path,cv2.IMREAD_GRAYSCALE)
        current_slice_set=[]
        for i in range(6):
            slice_path=cfg.ROOT_DIR+"input_data/slice_map/"+str(frame_number)+"_"+str(i)+".png"
            slice_image=cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            current_slice_set.append(slice_image)

        current_slice_set.append(density_image)
        total_data_set=np.concatenate(current_slice_set,axis=0).reshape(416,320,7)
        return total_data_set

    
    

    def get_slice_data_2(self, frame_number,root_path):
        density_path=root_path+"input_data/density_map/"+str(frame_number)+".png"
        density_image=cv2.imread(density_path,cv2.IMREAD_GRAYSCALE)
        current_slice_set=[]
        for i in range(5):
            slice_path=root_path+"input_data/slice_map/"+str(frame_number)+"_"+str(i)+".png"
            slice_image=cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            current_slice_set.append(slice_image)

        current_slice_set.append(density_image)
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




    def get_yolo_label(self):#이건 heading 방향 상관 없이 traning 한 것 즉, 여기서 rois는 왼쪽 위 포인트의 좌표값과 width와 angler값이다.
        anchors=self.read_anchors_file(cfg.ANCHOR_PATH)
        n_anchors = np.shape(anchors)[0]
        rois = ["[[86, 116,20,50]]"]#여기  col, row, width,height는 임으로 대충 정한값 같네.
        classes = ["[0]"]

        for rois, classes in zip(rois, classes):  # 하나하나 bounding box에 대해서
            rois = np.array(eval(rois), dtype=np.float32)  # eval을 통해서 string을 tuple로 만들고 numpy array로 만든다.
            classes = np.array(eval(classes), dtype=np.int32)
            raw_h = 400
            raw_w = 300

            label = np.zeros([cfg.GRID_H, cfg.GRID_W, n_anchors, 6], dtype=np.float32)#이거 틀림 완전 고쳐야 함 .
            for roi, cls in zip(rois, classes):  # 각 bounding box와 class정보인듯하다. 1개의 bounding box 인데.
                #print(roi)
                #print(cls)
                active_indxs = self.get_active_anchors(roi,anchors)  # 해당 프레임에서 gt와 특정 iou값을 넘어서는 anchor의 inx가 ative_index이다.
                #print(active_indxs)
                grid_x, grid_y = self.get_grid_cell(roi, raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)  # 해당 bd가 어느 grid에 위치 하는지
                print(grid_x, grid_y)

                for active_indx in active_indxs:  # 해당 gt와 많이 겹치는 각각의 anchor에 대해
                    anchor_label = self.roi2label(roi, anchors[active_indx], raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [cls], [1.0]))

        return label


    def get_yolo_label_1(self, obj_label_path):#이거 heading을 더하기 전에것을 사용.
        anchors=self.read_anchors_file(cfg.ANCHOR_PATH)
        n_anchors = np.shape(anchors)[0]
        obj_labels = [line for line in open(obj_label_path, 'r').readlines()]

        obj_label_list=[]
        for line in obj_labels:#주의 해야 할 것은 회전하기 전의 값을 기준으로 width와 height를 정하기 때문에 헷갈리지 않도록 주의 해야함.
            ret = line.split()
            center_x, center_y, width, height, angle = [float(i) for i in ret]#여기서 나오는 center_x, center_y값들은 lidar 즉 차량을 기준으로 한 값이기 때문에
            center_row=400-center_y
            center_col=300-center_x
            obj_label_list.append([[center_col, center_row, width, height,angle]])
        classes = ["[0]"]
        for rois, classes in zip(obj_label_list, classes):  # 하나하나 bounding box에 대해서
            rois = np.array(rois, dtype=np.float32)  # eval을 통해서 string을 tuple로 만들고 numpy array로 만든다.
            classes = np.array(eval(classes), dtype=np.int32)
            raw_h = 400
            raw_w = 300
            label = np.zeros([cfg.GRID_H, cfg.GRID_W, n_anchors, 6], dtype=np.float32)#이거 틀림
            for roi, cls in zip(rois, classes):  # 각 bounding box와 class정보인듯하다. 1개의 bounding box 인데.
                print(roi)
                active_indxs = self.get_active_anchors_2(roi,anchors)  # 해당 프레임에서 gt와 특정 iou값을 넘어서는 anchor의 inx가 ative_index이다.
                print("ative indxs",active_indxs)#여기서 나오는 anchor의 값은 회전하기 전의 anchor의 값이다.
                grid_x, grid_y = self.get_grid_cell_2(roi, raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)  # 해당 bd가 어느 grid에 위치 하는지
                print(grid_x, grid_y)

                for active_indx in active_indxs:  # 해당 gt와 많이 겹치는 각각의 anchor에 대해
                    anchor_label = self.roi2label_2(roi, anchors[active_indx], raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [cls], [1.0]))

        return label


    def get_yolo_label_2(self, obj_label_path):
        anchors=self.read_anchors_file(cfg.ANCHOR_PATH)
        n_anchors = np.shape(anchors)[0]
        obj_labels = [line for line in open(obj_label_path, 'r').readlines()]

        obj_label_list=[]
        for line in obj_labels:#주의 해야 할 것은 회전하기 전의 값을 기준으로 width와 height를 정하기 때문에 헷갈리지 않도록 주의 해야함.
            ret = line.split()
            center_x, center_y, width, height, angle = [float(i) for i in ret]#여기서 나오는 center_x, center_y값들은 lidar 즉 차량을 기준으로 한 값이기 때문에
            #여기서 나오는 angle값은 rect의 장축이 수직일 때를 0으로 보고 세계방향으로 돌아서 180도까지 가는 것으로 하고 왼쪽 오른쪽 구분은 하지 않는다.
            #즉 -10를 170로 표현되는 것이다. 값은 다르지만 사실상 visual시키면 그렇기 때문이다.
            #그리고 angle값을 그대로 사용하되 visual할 때에는 gt angle+90을 해서 아래와 같이 나오게 한다.
            #print(center_x, center_y, width, height, angle)  # 여기서 나오는 값의 기준은 장축이 수직일때를 0를 보고 한다.
            #box = cv2.boxPoints(((300 - center_x, 400 - center_y), (width, height), 90 + angle))
            #여기서 써야 하는 center의 값
            center_row=400-center_y
            center_col=300-center_x
            #print(center_col, center_row)
            #heading을 먹이기 전의 width, height이므로 50, 20 이다.
            #실제 이미지상에서는 20,50높이가 더 큰 사각형 형태인데
            obj_label_list.append([[center_col, center_row, width, height,angle]])

        classes = ["[0]"]
        label = np.zeros([cfg.GRID_H, cfg.GRID_W, n_anchors, 8], dtype=np.float32)  ######이거 틀림.
        for rois, classes in zip(obj_label_list, classes):  # 하나하나 bounding box에 대해서
            rois = np.array(rois, dtype=np.float32)  # eval을 통해서 string을 tuple로 만들고 numpy array로 만든다.
            classes = np.array(eval(classes), dtype=np.int32)
            raw_h = 400
            raw_w = 300
            for roi, cls in zip(rois, classes):  # 각 bounding box와 class정보인듯하다. 1개의 bounding box 인데.
                active_indxs = self.get_active_anchors_2(roi,anchors)  # 해당 프레임에서 gt와 특정 iou값을 넘어서는 anchor의 inx가 ative_index이다.
                #print(active_indxs)#여기서 나오는 anchor의 값은 회전하기 전의 anchor의 값이다.
                grid_x, grid_y = self.get_grid_cell_2(roi, raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)  # 해당 bd가 어느 grid에 위치 하는지
                #print(grid_x, grid_y)

                for active_indx in active_indxs:  # 해당 gt와 많이 겹치는 각각의 anchor에 대해
                    anchor_label = self.roi2label_2(roi, anchors[active_indx], raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)
                    cos_val=np.cos(roi[4]*np.pi/180.0)
                    sin_val=np.sin(roi[4]*np.pi/180.0)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [cls], [cos_val],[sin_val],[1.0]))#
                    #각도는 degree에서 radian으로 변경해서 넣기.

        return label



    def get_obj_label(self, obj_label_path):###새로운 type 이건 임시 버전
        anchors=self.read_anchors_file(cfg.ANCHOR_PATH)
        n_anchors = np.shape(anchors)[0]
        obj_labels = [line for line in open(obj_label_path, 'r').readlines()]

        obj_label_list=[]
        for line in obj_labels:#주의 해야 할 것은 회전하기 전의 값을 기준으로 width와 height를 정하기 때문에 헷갈리지 않도록 주의 해야함.
            ret = line.split()
            center_x, center_y, width, height, angle = [float(i) for i in ret]#여기서 나오는 center_x, center_y값들은 lidar 즉 차량을 기준으로 한 값이기 때문에
            obj_label_list.append([[center_x, center_y, width, height,angle]])
            #center_x 가 col 값이고 center_y값이 row 값이다.

        classes = [[0]]
        for rois, classes in zip(obj_label_list, classes):  # 하나하나 bounding box에 대해서
            rois = np.array(rois, dtype=np.float32)  # eval을 통해서 string을 tuple로 만들고 numpy array로 만든다.
            classes = np.array(classes, dtype=np.int32)
            raw_h = 640
            raw_w = 480
            label = np.zeros([cfg.GRID_H, cfg.GRID_W, n_anchors, 6], dtype=np.float32)
            for roi, cls in zip(rois, classes):  # 각 bounding box와 class정보인듯하다. 1개의 bounding box 인데.
                active_indxs = self.get_active_anchors_2(roi,anchors)  # 해당 프레임에서 gt와 특정 iou값을 넘어서는 anchor의 inx가 ative_index이다.
                print("active anchor : ",active_indxs)#여기서 나오는 anchor의 값은 회전하기 전의 anchor의 값이다.
                grid_x, grid_y = self.get_grid_cell_2(roi, raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)  # 해당 bd가 어느 grid에 위치 하는지
                print(grid_x, grid_y)

                for active_indx in active_indxs:  # 해당 gt와 많이 겹치는 각각의 anchor에 대해
                    anchor_label = self.roi2label_2(roi, anchors[active_indx], raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [cls], [1.0]))

        return label



    def get_obj_label_2(self, obj_label_path):##좀 더 genelize
        anchors=self.read_anchors_file(cfg.ANCHOR_PATH)
        n_anchors = np.shape(anchors)[0]
        obj_labels = [line for line in open(obj_label_path, 'r').readlines()]

        obj_label_list,classes=[],[]
        for line in obj_labels:#주의 해야 할 것은 회전하기 전의 값을 기준으로 width와 height를 정하기 때문에 헷갈리지 않도록 주의 해야함.
            ret = line.split()
            center_x, center_y, width, height, angle = [float(i) for i in ret]#여기서 나오는 center_x, center_y값들은 lidar 즉 차량을 기준으로 한 값이기 때문에
            obj_label_list.append([[center_x, center_y, width, height,angle]])
            #print("info, ",center_x, center_y, width, height,angle)
            classes.append([0])
            #center_x 가 col 값이고 center_y값이 row 값이다.

        label = np.zeros([cfg.GRID_H, cfg.GRID_W, n_anchors, 6], dtype=np.float32)
        for rois, classes in zip(obj_label_list, classes):  # 하나하나 bounding box에 대해서
            rois = np.array(rois, dtype=np.float32)  # eval을 통해서 string을 tuple로 만들고 numpy array로 만든다.
            classes = np.array(classes, dtype=np.int32)
            raw_h = 640
            raw_w = 480
            for roi, cls in zip(rois, classes):  # 각 bounding box와 class정보인듯하다. 1개의 bounding box 인데.
                active_indxs = self.get_active_anchors_2(roi,anchors)  # 해당 프레임에서 gt와 특정 iou값을 넘어서는 anchor의 inx가 ative_index이다.
                #print("active anchor : ",active_indxs)#여기서 나오는 anchor의 값은 회전하기 전의 anchor의 값이다.
                grid_x, grid_y = self.get_grid_cell_2(roi, raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)  # 해당 bd가 어느 grid에 위치 하는지
                #print(grid_x, grid_y)

                for active_indx in active_indxs:  # 해당 gt와 많이 겹치는 각각의 anchor에 대해
                    anchor_label = self.roi2label_2(roi, anchors[active_indx], raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [cls], [1.0]))
                    #print(grid_y, grid_x, label[grid_y, grid_x, active_indx])

        return label



    def get_obj_label_3(self, obj_label_path):##좀 더 genelize+with heading estimation
        anchors=self.read_anchors_file(cfg.ANCHOR_PATH)
        n_anchors = np.shape(anchors)[0]
        obj_labels = [line for line in open(obj_label_path, 'r').readlines()]

        obj_label_list,classes=[],[]
        for line in obj_labels:#주의 해야 할 것은 회전하기 전의 값을 기준으로 width와 height를 정하기 때문에 헷갈리지 않도록 주의 해야함.
            ret = line.split()
            center_x, center_y, width, height, angle = [float(i) for i in ret]#여기서 나오는 center_x, center_y값들은 lidar 즉 차량을 기준으로 한 값이기 때문에
            obj_label_list.append([[center_x, center_y, width, height,angle]])
            classes.append([0])

        label = np.zeros([cfg.GRID_H, cfg.GRID_W, n_anchors, 8], dtype=np.float32)
        for rois, classes in zip(obj_label_list, classes):  # 하나하나 bounding box에 대해서
            rois = np.array(rois, dtype=np.float32)  # eval을 통해서 string을 tuple로 만들고 numpy array로 만든다.
            classes = np.array(classes, dtype=np.int32)
            raw_h = 640
            raw_w = 480
            for roi, cls in zip(rois, classes):  # 각 bounding box와 class정보인듯하다. 1개의 bounding box 인데.
                active_indxs = self.get_active_anchors_2(roi,anchors)  # 해당 프레임에서 gt와 특정 iou값을 넘어서는 anchor의 inx가 ative_index이다.
                #print(active_indxs)
                grid_x, grid_y = self.get_grid_cell_2(roi, raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)  # 해당 bd가 어느 grid에 위치 하는지
                #print(grid_x,grid_y)
                cos_val, sin_val = np.cos(roi[4] * np.pi / 180.0), np.sin(roi[4] * np.pi / 180.0)
                #print(cos_val,sin_val)
                for active_indx in active_indxs:  # 해당 gt와 많이 겹치는 각각의 anchor에 대해
                    anchor_label = self.roi2label_2(roi, anchors[active_indx], raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [cos_val],[sin_val],[cls],[1.0]))#


        return label


    def get_obj_label_4(self, obj_label_path):##좀 더 genelize+with heading estimation for 400x300
        anchors=self.read_anchors_file(cfg.ANCHOR_PATH)
        n_anchors = np.shape(anchors)[0]
        obj_labels = [line for line in open(obj_label_path, 'r').readlines()]

        obj_label_list,classes=[],[]
        for line in obj_labels:#주의 해야 할 것은 회전하기 전의 값을 기준으로 width와 height를 정하기 때문에 헷갈리지 않도록 주의 해야함.
            ret = line.split()
            center_x, center_y, width, height, angle = [float(i) for i in ret]#여기서 나오는 center_x, center_y값들은 lidar 즉 차량을 기준으로 한 값이기 때문에
            obj_label_list.append([[center_x, center_y, width, height,angle]])
            classes.append([0])

        label = np.zeros([cfg.GRID_H, cfg.GRID_W, n_anchors, 8], dtype=np.float32)
        for rois, classes in zip(obj_label_list, classes):  # 하나하나 bounding box에 대해서
            rois = np.array(rois, dtype=np.float32)  # eval을 통해서 string을 tuple로 만들고 numpy array로 만든다.
            classes = np.array(classes, dtype=np.int32)
            raw_h = cfg.IMAGE_H
            raw_w = cfg.IMAGE_W
            for roi, cls in zip(rois, classes):  # 각 bounding box와 class정보인듯하다. 1개의 bounding box 인데.
                active_indxs = self.get_active_anchors_2(roi,anchors)  # 해당 프레임에서 gt와 특정 iou값을 넘어서는 anchor의 inx가 ative_index이다.
                #print(active_indxs)
                grid_x, grid_y = self.get_grid_cell_2(roi, raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)  # 해당 bd가 어느 grid에 위치 하는지
                #print(grid_x,grid_y)
                cos_val, sin_val = np.cos(roi[4] * np.pi / 180.0), np.sin(roi[4] * np.pi / 180.0)
                #print(cos_val,sin_val)
                for active_indx in active_indxs:  # 해당 gt와 많이 겹치는 각각의 anchor에 대해
                    anchor_label = self.roi2label_2(roi, anchors[active_indx], raw_w, raw_h, cfg.GRID_W, cfg.GRID_H)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [cos_val],[sin_val],[cls],[1.0]))


        return label


    def changle_angle_for_draw(self, width, height, angle):
        if (width < height):
            angle = angle - 180
        else:
            angle = angle - 90


        return angle









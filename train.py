from model_0822 import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main():
    train=YHMODEL()
    train.curb_detection_based_on_line()

if __name__=='__main__':
    main()


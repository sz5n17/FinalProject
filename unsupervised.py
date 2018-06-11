import cv2 as cv
import numpy as np
import os
import yaml
import sys
import math

def read_and_save(filepath):
    #Load
    print('Loading mission.yaml')
    mission = filepath + 'mission.yaml'
    with open(mission,'r') as stream:
        load_data = yaml.load(stream)
    for i in range(0,len(load_data)):
        if 'image' in load_data:
            image_filepath = load_data['image']['filepath']
    image_fullpath = filepath + image_filepath
    os.chdir(image_fullpath)
    name = np.loadtxt('image_name.txt',dtype=str)
    #Check if the data are in folder raw
    sub_path = image_fullpath.split(os.sep)
    for i in range(1,(len(sub_path))):
        if sub_path[i]=='raw':
            flag_f = True
    if flag_f == False:
        print('Check folder structure contains "raw"')
    #Save
    sub_out = sub_path
    outpath = sub_out[0]
    proc_flag = 0
    for i in range(1,len(sub_path)):
        if sub_path[i] == 'raw':
            sub_out[i] = 'processed'
            proc_flag = 1
        else:
            sub_out[i] = sub_path[i]
        outpath = outpath + os.sep + sub_out[i]
    if proc_flag==1:
        if os.path.isdir(outpath)==0:
            try:
                os.mkdir(outpath)
            except Exception as e:
                print("Warning:",e)
        for i in range(len(name)):
            img = cv.imread(name[i])
            img = swap_channels(img)
            path_new = outpath + os.sep + name[i]
            cv.imwrite(path_new,img)

def swap_channels(img):
    b,g,r = cv.split(img)
    img_swapped = cv.merge((g,b,r))
    return img_swapped

def syntax_error():
    print("unsupervised.py <options>")
    print(" -i <path to mission.yaml>")
    return -1

if __name__ =='__main__':
    if (int(len(sys.argv)))<2:
        print('Error: not enough arguments')
        syntax_error()
    else:
        for i in range(int(math.ceil(len(sys.argv)))):
            option=sys.argv[i]
            if option =="-i":
                filepath = sys.argv[i+1]
                flag_i = True
                if flag_i == False:
                    print('Error: incorrect use')
                    syntax_error()
                else:
                    sys.exit(read_and_save(filepath))
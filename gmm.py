#  Library
import cv2 as cv
import numpy as np
import os
import yaml
import sys
import math


"""
    Mission Loading, Processing and Results Saving Function:
    //input  filepath           location of 'mission.yaml' in OS (string)
    //param  load_data          loaded mission (object)
             image_fullpath     location of images in OS (string)
             name               image name (string)
             outpath            location of outputs in OS (string)
             img                image to be clustered (object)
             flag               cluster number (integer)
             cluster            cluster number (char)
             path_new           locaiton of output images in OS (string)
"""
def read_and_save(filepath):
    # Load mission
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
    # Check if the data are in folder raw
    sub_path = image_fullpath.split(os.sep)
    for i in range(1,(len(sub_path))):
        if sub_path[i]=='raw':
            flag_f = True
    if flag_f == False:
        print('Check folder structure contains "raw"')
    # Processing and Saving
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
            flag = gmm(img)
            cluster = chr((flag + 48))
            path_new = outpath + os.sep + cluster + os.sep+ name[i]
            cv.imwrite(path_new,img)

"""
    Gaussian Mixture Model Clustering funciton (GMM):
    //input   img        image to be identified (object)
    //param   b, g, r    pixels values (matrix)
              n_pixels   number of pixels (scalar)
              T          truncation level
    //param   x:      Obeservations (1 x 6 vector)
              x(0)       average value of Blue channels (scalar)
              x(1)       average value of Green channels (scalar)
              x(2)       average value of Red channels (scalar)
              x(3)       entropy of Blue color (scalar)
              x(4)       entropy of Greeen color (scalar)
              x(5)       entropy of Red color (scalar)
    //output  flag       cluster number (integer)
"""
def gmm(img):
    b,g,r = cv.split(img)
    n_pixels = np.size(b)
    # Average value of BGR
    x(0) = sum(sum(b))/n_pixels
    x(1) = sum(sum(g))/n_pixels
    x(2)= sum(sum(r))/n_pixels
    # Entropy of BGR
    x(3) = entropy(b)
    X(4) = entropy(g)
    x(5) = entropy(r)
    
    T = 3
    return flag

"""
    Entropy Calculation Funciton:
    //input   channel       color channel to be computed (matrix)
    //param   count         number of pixels for different values (vector)
              y          pixel values (vector)
              total         total number of pixels (scalar)
              p             Bayes Posterior Probability (vector)
              logp          logrithm of p with base 2 (vector)
              entropy_step  entropy of each value (scalar)
    //ouput   entropy       total entropy (scalar)
"""
def entropy(channel):
    [count, y] = imhist(channel)
    total = sum(count)
    entropy = 0
    for i = 1:256
      p(i) = count(i)/total;
      if p(i) != 0:
        logp(i)=log2(p(i))
        entropy_step = -p(i)*logp(i)
        entropy = entropy + entropy_step
    return entropy


"""
    Syntax Error Function:
    Print Info when error of options detected
"""
def syntax_error():
    print("unsupervised.py <options>")
    print(" -i <path to mission.yaml>")
    return -1

"""
    Main Function
    Call it in cmd by 'gmm.py -i <path to mission.yaml>'
"""
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
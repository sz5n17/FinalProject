#  Library
import cv2 as cv
import numpy as np
import os
import yaml
import sys
import math
import random


"""
    Mission Loading, Processing and Results Saving Function:
    //input  filepath           location of 'mission.yaml' in OS (string)
    //param  load_data          loaded mission (object)
             image_fullpath     location of images in OS (string)
             name               name list of images (string vector)
             outpath            location of outputs in OS (string)
             x                  obeservation (N x 6 float matrix)
             cluster            cluster number (1 x N char vector)
             img                image to be identified (object)
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
    # Cluster and Save
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
        #Cluster using GMM
        x = Getfeature(img, name)    
        cluster = GMM(x)
        #Save into different folders
        for i in range(len(name)):
            img = cv.imread(name[i])
            path_new = outpath + os.sep + cluster[i] + os.sep+ name[i]
            cv.imwrite(path_new,img)


    """
    Get Feature (Oberservation) Function:
    //input   img        image to be identified (object)
              name       name list of images (string vector)
    //param   b, g, r    pixels values (matrix)
              n_pixels   number of pixels (int)
    //output  x:      Obeservation  (N x 6 float matrix)
              x[i,0]       normalized average value of Blue channels (float)
              x[i,1]       normalized average value of Green channels (float)
              x[i,2]       normalized average value of Red channels (float)
              x[i,3]       entropy of Blue color (float)
              x[i,4]       entropy of Greeen color (float)
              x[i,5]       entropy of Red color (float)
    """
    def Getfeature(img, name):
        x = np.zeros((len(name),6))
        for i in range(len(name)):
            img = cv.imread(name[i])
            b,g,r = cv.split(img)
            n_pixels = np.size(np.size(b))
            # Average value of BGR
            x[i,0] = (sum(sum(b))/n_pixels)./255
            x[i,1] = (sum(sum(g))/n_pixels)./255
            x[i,2] = (sum(sum(r))/n_pixels)./255
            # Entropy of BGR
            x[i,3] = entropy(b)
            x[i,4] = entropy(g)
            x[i,5] = entropy(r)
        return x
    
        # Entropy Calculation Funciton:
        # //input   channel       color channel to be computed (matrix)
        # //param   count         number of pixels for different values (vector)
        #           y             pixel values (vector)
        #           total         total number of pixels (scalar)
        #           p             Bayes Posterior Probability (vector)
        #           logp          logrithm of p with base 2 (vector)
        #           entropy_step  entropy of each value (scalar)
        # //ouput   entropy       total entropy (scalar)
        def entropy(channel):
            [count, y] = np.histogram(channel,bins=np.arange(255))
            total = sum(count)
            entropy = 0
            p = np.zeros(256)
            for i in range(256):
              p[i] = count[i]/total
              if p[i] != 0:
                logp[i]=log2(p[i])
                entropy_step = - p[i]*logp[i]
            entropy = entropy + entropy_step
        return entropy


    """
    Gaussian Mixture Model Clustering funciton (GMM):
    //input   x          observation (N x 6 float matrix)
              T          truncation level (scalar)
              miu        centroids of clusters (T x 6 float matrix)
              sigma      standard deviation (6 x 6 x T float matrix)
              pi         cluster weight (1 x T float vector)
              flag       cluster number (1 x N int vector)
    //output  cluster    cluster number (1 x N char vector)
    """
    def GMM(x):
        # Initialize
        flag = np.zeros_like(x)
        T = 3
        [miu, sigma, pi] = param_init(x,T)
        # Update
        for k in range(T):
            z[i,k] = pi * Gaussian_prob(x[i],miu[k,:],sigma[:,:,k],T)
            
            
        cluster = chr((flag + 48))
        return cluster
    
        # Initialize miu, sigma, pi
        def param_init(x,T):
            miu = random.sample(x,T)
            sigma = np.zeros((6,6,T))
            pi = np.zeros(T)
        return [miu, sigma, pi]

        # Gaussian Posterior Probability
        def Gaussian_prob(x_i, miu_k, sigma_k)
            xshift = x_i - np.ones((N,1)) * miu_k
            inv_sigma_k = inv(siamg_k)
            tmp = sum(sum((xshift * inv_sigma_k) * xshift))
            coef = (2*np.pi)^(-3) * sqrt(np.linalg.det(inv_sigma_k))
            prob = coef * np.exp(-0.5 * tmp)
            return prob
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
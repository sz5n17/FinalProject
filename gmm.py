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
             cluster            cluster number (N x 1 char vector)
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
        cluster = GMM(x, len(name))
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
            n_pixels = np.size(b)
            # Average value of BGR
            x[i,0] = (np.sum(b)/n_pixels)./255
            x[i,1] = (np.sum(g)/n_pixels)./255
            x[i,2] = (np.sum(r)/n_pixels)./255
            # Entropy of BGR
            x[i,3] = entropy(b)
            x[i,4] = entropy(g)
            x[i,5] = entropy(r)
        return x
    
        # Entropy Calculation Funciton:
        # //input   channel       color channel to be computed (matrix)
        # //param   count         number of pixels for different values (vector)
        #           y             pixel values (vector)
        #           total         total number of pixels (int)
        #           p             Bayes Posterior Probability (vector)
        #           logp          logrithm of p with base 2 (vector)
        #           entropy_step  entropy of each value (float)
        # //ouput   entropy       total entropy (float)
        def entropy(channel):
            [count, y] = np.histogram(channel,bins=np.arange(255))
            total = np.sum(count)
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
    //param   T          truncation level (scalar)
              N          number of observations (int)
              miu        centroids of clusters (T x 6 float matrix)
              sigma      standard deviation (6 x 6 x T float matrix)
              pi         cluster weight (1 x T float vector)
              flag       cluster number (N x 1 int vector)
    //output  cluster    cluster number (N x 1 char vector)
    """
    def GMM(x):
        # Initialize
        [N, T] = np.shape(x)
        N = int(N)
        T = int(T)
        flag = np.zeros((N,1))
        [miu, sigma, pi] = param_init(x)
        tolerance = 1e-15
        Lprev = -float("inf")
        while(Lprev > tolerance):
            # Update
            z = pi * Gaussian_prob(x,miu,sigma)
        cluster = chr((flag + 48))
        return cluster
    
        # Initialize miu, sigma, pi using k-means method
        # //param    x_k: observations in cluster k
        def param_init(x):
            [N, T] = np.shape(x)
            N = int(N)
            T = int(T)
            miu = random.sample(x,T)
            sigma = np.zeros((6,6,T))
            pi = np.zeros(T)
            distant = np.zeros_like(x)
            for i in range(T):
                # calculate distant
                tmp = x - np.dot(np.ones((N,1)),miu[i])
                tmp = tmp * tmp
                tmp = np.sum(tmp,axis=1)
                distant[:,i] = tmp.reshape(N,1)
                cluster = np.argmin(distant,axis=1)
                cluster = cluster.reshape(N,1)
            for i in range(T):
                x_k = x[cluster == i, :]
                pi[i] = int(np.size(x_k,axis=0))/N
                sigma[:,:,i] = np.cov(x_i)
            return [miu, sigma, pi]

        # Gaussian Posterior Probability
        # //param    inv_sigma: inverse matrix of sigma
        def Gaussian_prob(x, miu, sigma):
            [N, T] = np.shape(x)
            N = int(N)
            T = int(T)
            prob = np.zeros_like(x)
            for k in range(T):
                x_shift = x - np.dot(np.ones((N,1)),miu[k])
                inv_sigma = np.linalg.inv(siamg[:,:,k])
                tmp = np.sum((np.dot(x_shift,inv_sigma) * xshift),axis=1)
                coef = (2*np.pi)^(-T/2) * np.sqrt(np.linalg.det(inv_sigma_k))
                prob[:,k] = coef * np.exp(-0.5 * tmp)
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
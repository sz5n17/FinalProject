#  Library
import cv2 as cv
import numpy as np
import os
import yaml
import sys
import math
import random

"""Function Definitions"""

"""Feature Getting Function"""
"""
//Child-function:       Entropy Calculation Funciton
//input   channel       color channel to be computed (matrix)
//param   count         number of pixels for different values (vector)
          y             pixel values (vector)
          total         total number of pixels (int)
          p             Bayes Posterior Probability (vector)
          logp          logrithm of p with base 2 (vector)
          entropy_step  entropy of each value (float)
//ouput   entropy       total entropy (float)
"""
def entropy(channel):
    [count, y] = np.histogram(channel,bins=np.arange(257))
    total = float(np.sum(count))
    entropy = 0.0
    p = np.zeros(256)
    logp = np.zeros_like(p)
    for i in range(256):
        p[i] = float(count[i])/total
        if p[i] != 0:
            logp[i]= np.log2(p[i])
            entropy_step = - p[i]*logp[i]
            entropy = entropy + entropy_step
    entropy = entropy/np.log2(256.0)
    return entropy
"""
//Parent function:   Getfeature function
//input   name       name list of images (string vector)
//param   img        image to be identified (object)
          b, g, r    pixels values (matrix)
          n_pixels   number of pixels (int)
//output  x:         obeservation  (N x 6 float matrix)
          x[i,0]     normalized average value of Blue channels (float)
          x[i,1]     normalized average value of Green channels (float)
          x[i,2]     normalized average value of Red channels (float)
          x[i,3]     normalized entropy of Blue color (float)
          x[i,4]     normalized entropy of Greeen color (float)
          x[i,5]     normalized entropy of Red color (float)
"""
def Getfeature(name):
    x = np.ones((len(name),6))
    for i in range(len(name)):
        img = cv.imread(name[i])
        b,g,r = cv.split(img)
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        n_pixels = float(np.size(b))
        # Average value of BGR
        x[i,0] = (np.sum(b)/n_pixels)/255.0*1.5
        x[i,1] = (np.sum(g)/n_pixels)/255.0*1.5
        x[i,2] = (np.sum(r)/n_pixels)/255.0*1.5
        # Entropy of BGR
        x[i,3] = entropy(b)*1.5
        x[i,4] = entropy(g)*1.5
        x[i,5] = entropy(r)*1.5
    return x


"""GMM Clustering Funciton"""
"""
//Child-function A:   Initialize miu, sigma, pi using k-means method
//input    x          obeservation  (N x 6 float matrix)
           T          truncation level (int)
//param    D          dimension of a single observation (int)
           N          number of observations (int)
           x_k        observations in cluster k
           cluster    cluster number (N x 1 int vector)
           distance   distance between observations and centroids (N x T float matrix)
//output   miu        centroids of clusters (T x 6 float matrix)
           sigma      standard deviation (6 x 6 x T float matrix)
           pi         cluster weight (1 x T float vector)
"""
def param_init(x,T):
    [N, D] = np.shape(x)
    N = int(N)
    D = int(D)
    miu = random.sample(x,T)
    print("miu")
    print(miu)
    sigma = np.zeros((6,6,T))
    pi = np.zeros(T)
    distance = np.zeros((N,T))
    for k in range(T):
        # calculate distant
        tmp = x - np.dot(np.ones((N,1)),miu[k].reshape(1,D))
        tmp = tmp * tmp
        tmp = np.sum(tmp,axis=1)
        distance[:,k] = tmp.reshape(N)
    cluster = np.argmin(distance,axis=1)
    cluster = cluster.reshape(N)
    for k in range(T):
        x_k = x[cluster == k, :]
        print("k=")
        print(k)
        print(x_k)
        pi[k] = float(np.size(x_k,axis=0))/N
        sigma[:,:,k] = np.cov(x_k,rowvar=False)
    return [miu, sigma, pi]
"""
//Child-function B:   Gaussian Posterior Probability
//input    T          truncation level (int)
           miu        centroids of clusters (T x 6 float matrix)
           sigma      standard deviation (6 x 6 x T float matrix)
           pi         cluster weight (1 x T float vector)
//param    D          dimension of a single observation (int)
           N          number of observations (int)
//output   prob       Gaussian posterior probability
"""
def Gaussian_prob(x, miu, sigma, T):
    [N, D] = np.shape(x)
    N = int(N)
    D = int(D)
    prob = np.zeros((N,T))
    for k in range(T):
        x_shift = x - np.dot(np.ones((N,1)),miu[k].reshape(1,D))
        inv_sigma = np.linalg.inv(sigma[:,:,k])
        inv_sigma.dtype = 'float'
        print(inv_sigma)
        tmp = np.sum((np.dot(x_shift,inv_sigma) * x_shift),axis=1)
        coef = np.power((2*np.pi),(-float(T)/2)) * np.sqrt(np.linalg.det(inv_sigma))
        prob[:,k] = coef * np.exp(-0.5 * tmp)
    return prob
"""
Gaussian Mixture Model Clustering funciton (GMM):
//input   x          observation (N x 6 float matrix)
//param   T          truncation level (int)
          D          dimension of a single observation (int)
          N          number of observations (int)
          miu        centroids of clusters (T x 6 float matrix)
          sigma      standard deviation (6 x 6 x T float matrix)
          pi         cluster weight (1 x T float vector)
          tolerance  threshold for convergence (float)
          p          Gaussian posterior probility (N x T float matrix)
          z          latent variable (prob of i-th observation by k-th cluster) (N x T float matrix)
          Nk         sum of latent variable (1 x T float vector)
          flag       cluster number (N x 1 int vector)
          cluster    cluster number (N x 1 string vector)
//output  model      all model informaiton (object)
"""
def GMM(x):
    # Initialize
    [N, D] = np.shape(x)
    N = int(N)
    D = int(D)
    flag = np.zeros((N,1))
    cluster = np.zeros_like(flag)
    T = 3
    [miu, sigma, pi] = param_init(x,T)
    tolerance = 1e-15
    Lprev = -float("inf")
    while(True):
        # Estimation
        p = Gaussian_prob(x, miu, sigma, T)
        z = np.dot(np.ones((N,1)),pi.reshape(1,T)) * p
        z = z / np.dot(np.sum(z,axis=1).reshape(N,1), np.ones((1,T)))
        # Maximization and updating
        Nk = np.sum(z,axis=0)
        # Weights
        pi = Nk / N
        # Centroids
        miu = np.dot(np.diag(1/Nk),np.dot(z.T,x))
        # Sigma
        for k in range(T):
            x_shift = x - np.dot(np.ones((N,1)),miu[k].reshape(1,D))
            tmp = np.dot(np.diag(z[:,k]),x_shift)
            tmp = np.dot(x_shift.T,tmp)
            sigma[:,:,k] = tmp/Nk[k]
        L = np.sum(np.log(np.dot(p, pi.reshape(T,1))))
        if L - Lprev < tolerance:
            break
        Lprev = L
    # Once convergencing, return model
    model = []
    model.x = x
    model.T = T
    model.miu = miu
    model.sigma = sigma
    model.p = Gaussian_prob(model.x, model.miu, model.sigma, model.T)
    flag = np.argmax(model.p,axis=1)
    for i in range(len(flag)):
        cluster[i] = str(flag[i])
    model.cluster = cluster
    return model
   

     
"""
    Mission Loading, Processing and Results Saving Function:
    //input  filepath           location of 'mission.yaml' in OS (string)
    //param  load_data          loaded mission (object)
             image_fullpath     location of images in OS (string)
             name               name list of images (string vector)
             outpath            location of outputs in OS (string)
             x                  obeservation (N x 6 float matrix)
             cluster            cluster number (N x 1 string vector)
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
        x = Getfeature(name)
        cluster = VDP(x)
        #Save into different folders
        for i in range(len(name)):
            img = cv.imread(name[i])
            path_new = outpath + os.sep + cluster[i] + os.sep+ name[i]
            cv.imwrite(path_new,img)


"""
    Syntax Error Function:
    Print Info when error of options detected
"""
def syntax_error():
    print("gmm.py <options>")
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
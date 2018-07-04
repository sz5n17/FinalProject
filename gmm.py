#  Library
import cv2 as cv
import numpy as np
import scipy
import os
import yaml
import sys
import math
import random
import csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pdb

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
        x[i,0] = (np.sum(b)/n_pixels)/255.0
        x[i,1] = (np.sum(g)/n_pixels)/255.0
        x[i,2] = (np.sum(r)/n_pixels)/255.0
        # Entropy of BGR
        x[i,3] = entropy(b)
        x[i,4] = entropy(g)
        x[i,5] = entropy(r)
    return x


"""GMM Clustering Funciton"""
"""
//Child-function:        Calculate Wk
//input
//param
//output
"""
def Wk_calc(x,cluster_number):
    km = KMeans(cluster_number).fit(x)
    centers = km.cluster_centers_
    labels = km.labels_
    Wk = 0
    for k in range(len(centers[:,0])):
        tmp = x[labels==k,:]
        N_tmp = len(tmp[:,0])
        disp = tmp - centers[k]
        disp = disp * disp
        sum = np.sum(disp)
        Wk_tmp = sum/(N_tmp*2)
        Wk = Wk + Wk_tmp
    return Wk
"""
//Child-function:        Get optimum cluster number using Gap Statistic
//input     max_cluster    Maximum of cluster number (int)
//param     gap
            refWk
            x_min
            x_max
            Wk
//output    T              optimum cluster number (int)
"""
def gap_statistic(x,nrefs,max_cluster):
    gap = np.zeros(max_cluster)
    sd = np.zeros(max_cluster)
    for index,k in enumerate(range(1,max_cluster)):
        refWk = np.zeros(nrefs)
        x_min = np.amin(x)
        x_max = np.amax(x)
        for i in range(nrefs):
            # Create new random reference set
            randomReference = (x_max - x_min) * np.random.random_sample(size=x.shape) + x_min * np.ones(x.shape)
            # Fit to it
            refWk[i] = Wk_calc(randomReference,cluster_number=k)
        Wk = Wk_calc(x,cluster_number=k)
        gap[index] = np.mean(np.log(refWk)) - np.log(Wk)
        sd[index] = np.std(np.log(refWk))
    sd = np.sqrt(1+1/nrefs)*sd
    T = np.argmax(gap) + 1
    return T
    
"""
//Child-function:   Initialize miu, sigma, pi using k-means method
//input    x          obeservation  (N x D float matrix)
           T          optimum cluster number (int)
//param    D          dimension of a single observation (int)
           N          number of observations (int)
           x_k        observations in cluster k
           cluster    cluster number (N x 1 int vector)
           distance   distance between observations and centroids (N x T float matrix)
//output   miu        centroids of clusters (T x D float matrix)
           sigma      standard deviation (D x D x T float matrix)
           pi         cluster weight (1 x T float vector)
"""
def param_init(x,T):
    if len(np.shape(x))>1:
        [N, D] = np.shape(x)
        N = int(N)
        D = int(D)
    else:
        N = len(x)
        D = 1
    miu = random.sample(x,T)
    sigma = np.zeros((D,D,T))
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
        pi[k] = float(np.size(x_k,axis=0))/N
        sigma[:,:,k] = np.cov(x_k,rowvar=False)
    return [miu, sigma, pi]
"""
//Child-function:   Gaussian Posterior Probability
//input    T          optimum cluster number (int)
           miu        centroids of clusters (T x D float matrix)
           sigma      standard deviation (D x D x T float matrix)
           pi         cluster weight (1 x T float vector)
//param    D          dimension of a single observation (int)
           N          number of observations (int)
//output   prob       Gaussian posterior probability
"""
def Gaussian_prob(x, miu, sigma, T):
    if len(np.shape(x))>1:
        [N, D] = np.shape(x)
        N = int(N)
        D = int(D)
    else: 
        N = len(x)
        D = 1
    prob = np.zeros((N,T))
    for k in range(T):
        x_shift = x - np.dot(np.ones((N,1)),miu[k].reshape(1,D))
        inv_sigma = np.linalg.inv(sigma[:,:,k])
        inv_sigma.dtype = 'float'
        tmp = np.sum((np.dot(x_shift,inv_sigma) * x_shift),axis=1)
        coef = np.power((2*np.pi),(-float(T)/2)) * np.sqrt(np.linalg.det(inv_sigma))
        prob[:,k] = coef * np.exp(-0.5 * tmp)
    return prob
"""
Gaussian Mixture Model Clustering funciton (GMM):
//input   x          observation (N x D float matrix)
//param   D          dimension of a single observation (int)
          N          number of observations (int)
          miu        centroids of clusters (T x D float matrix)
          sigma      standard deviation (D x D x T float matrix)
          pi         cluster weight (1 x T float vector)
          tolerance  threshold for convergence (float)
          p          Gaussian posterior probility (N x T float matrix)
          z          latent variable (prob of i-th observation by k-th cluster) (N x T float matrix)
          Nk         sum of latent variable (1 x T float vector)
          flag       cluster number (N x 1 int vector)
//output  cluster    cluster number (N x 1 string vector)
          T          optimum cluster number (int)
"""
def GMM(x):
    # Initialize
    if len(np.shape(x))>1:
        [N, D] = np.shape(x)
        N = int(N)
        D = int(D)
    else:
        N = len(x)
        D = 1
    T = gap_statistic(x,nrefs=20,max_cluster=10)
    cluster = np.zeros((N,1))
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
    # Once convergencing
    p = Gaussian_prob(x, miu, sigma, T)
    cluster = np.argmax(p,axis=1)
    """.astype(str)"""
    return cluster, T
   

     
"""
    Mission Loading, Processing and Results Saving Function:
    //input  filepath           location of 'mission.yaml' in OS (string)
    //param  load_data          loaded mission (object)
             image_fullpath     location of images in OS (string)
             name               name list of images (string vector)
             outpath            location of outputs in OS (string)
             x                  obeservation (N x 6 float matrix)
             T                  optimum cluster number (int)
             cluster            cluster number (N x 1 string vector)
             img                image to be identified (object)
             path_new           locaiton of output images in OS (string)
"""
def read_and_save(filepath):
    # Load mission
    print('Loading mission.yaml')
    filepath = str(filepath)
    mission = filepath + 'mission.yaml'
    with open(mission,'r') as stream:
        load_data = yaml.load(stream)
    for i in range(0,len(load_data)):
        if 'image' in load_data:
            image_filepath = load_data['image']['filepath']
    image_fullpath = filepath + image_filepath
    os.chdir(image_fullpath)
    with open('FileTime.csv','r') as filetime:
        reader = csv.DictReader(filetime)
        index = [row['index'] for row in reader]
        reader = csv.DictReader(filetime)
        time = [row['time'] for row in reader]
    name = np.loadtxt('image_name.txt',dtype=str)
    # Check if the data are in folder raw
    sub_path = image_fullpath.split(os.sep)
    flag_f = False
    for i in range(1,(len(sub_path))):
        if sub_path[i]=='raw':
            flag_f = True
    if flag_f == False:
        print('Check folder structure contains "raw"')
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

    #Clustering using GMM
    x = Getfeature(name)
    #np.savetxt('data.txt',x,delimiter=' ',newline='\n')
    #Perform PCA
    pca = PCA(n_components='mle',svd_solver='full')
    y = pca.fit_transform(x)
    cluster, T = GMM(y[:,0:2])
    count, cluster_number = np.histogram(cluster,bins=np.arange(T+1))
    cluster = cluster.astype(str) 
    print(count, cluster_number)
    cluster = cluster.astype(str)
    with open('output.csv','w') as output:
        writer = csv.writer(output)
        writer.writerow(['index','name','cluster','time'])
        for i in range(len(name)):
            writer.writerow([ index[i],name[i],cluster[i] ])
    """
    #Save into different folders
    for i in range(len(name)):
        img = cv.imread(name[i])
        path_tmp = outpath + os.sep + cluster[i]
        if os.path.isdir(path_tmp)==0:
            try:
                os.mkdir(path_tmp)
            except Exception as e:
                print("Warning:",e)
        path_new = path_tmp + os.sep + name[i]
        cv.imwrite(path_new,img)
    """

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
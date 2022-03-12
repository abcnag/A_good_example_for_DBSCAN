'''
type of points in DBSCAN :
1. Core : Points that contain at least a number of points with a definite radius
2. Border : Exactly the opposite of cores
3. Outline : Points that are not nuclei but do not hold any number of points with a certain radius 
-----
Cluster : Set of Cores that can be in each other
'''

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


X,Y = make_blobs(1000 , centers=([3,3],[5,5],[8,8]))
model = DBSCAN(eps=0.3,                # radius
               min_samples=7,          # number of minimum of sample at radius 
               algorithm='brute',      # algorithm (ball_tree , kd_tree , brute , auto)
               metric='euclidean'      # type of distance  
               )
model.fit(X)

y_pre = model.labels_
plt.scatter(x=X[:,0],y=X[:,1],c=y_pre)
plt.show()
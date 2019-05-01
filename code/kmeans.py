''' 
 Code for analysing Preferential Looking task used for the
 assessment of social motivation using gaze
 Copyright (C) 2018 Rahul Bishain and Sharat Chandran
 
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.

 Further, the main program (included in folder named 'code') calls the 
 gaze tracking algorithm iTracker (incuded in the folder named 'csail')
 which is governed by its own licensing terms. Please refer to 
 https://github.com/CSAILVision/GazeCapture for its license agreement
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 
  Created on: Apr, 2018
      Author: Rahul Bishain
    Filename: kmeans.py
'''


from sklearn.cluster import KMeans
import os
import numpy as np

srcdir = "data/input/"
srcfile = "output_file.csv"
srcoutfile1 = "output_with_clustering.csv"
'''
screen_center = 2
right_boundary = 20
left_boundary = -15 
top_boundary = 3
bottom_boundary = -15
'''

def getClusterPreds(srcdir):
#outString = []

    for folder in os.listdir(srcdir):
        infile = srcdir+folder+"/"+srcfile
        if os.path.isfile(infile):
            print("processing: folder",folder)
            centroids,datanew = getMeans(infile)
#            outString.append(folder+","+",".join([str(x) for x in centroids])+"\n")
        else:
            print(" output file missing for folder",folder)

        f1 = open(srcdir+folder+"/"+srcoutfile1,"w")
        for data in datanew:
            f1.write(data)
        f1.close()

# add processing - if center doesnt lie between the two centroids, then provide only one prediction"


def getMeans(infile):
    f1 = open(infile,"r")
    data = f1.readlines()
    data = [ ele.strip() for ele in data]
    f1.close()

    X = []
    for row in data:
        tempRow = list(map(float,row.split()))
        '''
        if tempRow[-3] < left_boundary or tempRow[-3] > right_boundary or tempRow[-2] > top_boundary or tempRow[-2] < bottom_boundary:
            gazeOut+=1
            continue
        '''
        X.append(tempRow[-3])
    X = np.array(X)
    X = X.reshape(-1,1)

    kmeans = KMeans(n_clusters=2).fit(X)
    centroids = kmeans.cluster_centers_
    centroids = list((centroids[0,0],centroids[1,0]))
    centroids.sort()
    labelmap = {kmeans.predict(centroids[0])[0]:0, kmeans.predict(centroids[1])[0]:1}
    predLabel = []
    for row in data:
        tempRow = list(map(float,row.split()))
        '''
        if tempRow[-3] < left_boundary or tempRow[-3] > right_boundary or tempRow[-2] > top_boundary or tempRow[-2] < bottom_boundary:
            gazeOut+=1
            predLabel.append(2)
            continue
        '''
        predLabel.append(labelmap[kmeans.predict(tempRow[-3])[0]])
#    print(labelmap)
#    print(centroids) 
    
    for i in range(len(data)):
        temp = data[i]
        temp = temp.split()
        temp1 = ",".join(temp[0:-1])
        data[i] = temp1+","+str(int(predLabel[i]))+"\n"
    
    return (centroids,data)

if __name__ == "__main__":
    getClusterPreds(srcdir)

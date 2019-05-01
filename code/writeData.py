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
    Filename: writeData.py
'''


import numpy as np
import cv2

def write(fil_path,i,shape,lEye_img,rEye_img,lEye2save,rEye2save,lAvg,rAvg,lCG,rCG,outFile):
    mCG = (lCG+rCG)/2
    cv2.imwrite("/home/rahulb/PhD/iTrack/72_1502269024071_looking/images/eyes/"+str(i)+"_l.jpg",lEye2save)
    cv2.imwrite("/home/rahulb/PhD/iTrack/72_1502269024071_looking/images/eyes/"+str(i)+"_r.jpg",rEye2save)
    cv2.imwrite(fil_path+"eyes/"+str(i)+"_l.jpg",lEye_img)
    cv2.imwrite(fil_path+"eyes/"+str(i)+"_r.jpg",rEye_img)
    i = np.array([[i]])
    # print(lAvg.shape,lCG.shape,i)
#cv2.imshow("leftEye",lEye2save)
#cv2.waitKey(0)
#assert False
    arr = np.concatenate((i,shape.reshape(-1,1),lAvg,rAvg,lCG,rCG,mCG))
    # print(arr.shape)

    np.savetxt(outFile, arr.T, delimiter=" ")

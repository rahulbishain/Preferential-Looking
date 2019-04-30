''' 
 Code for analysing Preferential Looking task used for the
 assessment of social motivation using gaze
 Copyright (C) 2018 Rahul Bishain and Sharat Chandran
 
 This file can be redistributed and/or modified under the terms of the 
 GNU General Public License as published by the Free Software 
 Foundation, either version 3 of the License, or (at your option) 
 any later version. Further, the main program utilizes the 
 gaze tracking algorithm iTracker (incuded in the folder named 'csail')
 which is governed by its own licensing terms. Please refer to 
 https://github.com/CSAILVision/GazeCapture for its license agreement
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.


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

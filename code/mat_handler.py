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
    Filename: mat_handler.py
'''


import scipy.io as sio
import numpy as np

def save_mat(np_array):
    print("array size after extraction ",np_array.shape)         
    temp_dict={}                                                       
    temp_dict['frameIndex']=np_array[:,0:1].astype(np.int32)     
    temp_dict['labelDotXCam']=np_array[:,1:2].astype(np.double)  
    temp_dict['labelDotYCam']=np_array[:,2:3].astype(np.double)  
    temp_dict['labelFaceGrid']=np_array[:,3:7].astype(np.double) 
    temp_dict['labelRecNum']=np_array[:,7:8].astype(np.int16)    
    temp_dict['labelTest']=np_array[:,8:9].astype(bool)          
    temp_dict['labelTrain']=np_array[:,9:10].astype(bool)        
    temp_dict['labelVal']=np_array[:,10:].astype(bool)           
    # print("dict size ",len(temp_dict))
    sio.savemat('metadata',temp_dict)

def load_mat(file_path):
    temp_dict = sio.loadmat(file_path+'metadata.mat')
    np_array = np.concatenate((temp_dict['frameIndex'],temp_dict['labelFaceGrid'],temp_dict['labelRecNum']),1)
    return np_array
    # print("shape of loaded data ",np_array.shape)


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
    Filename: get_crops.py
'''


from get_landmarks import get_landmark_array
import dlib
import cv2
from cropfun import crop
import os
import numpy as np
import scipy.io as sio
from sys_call import sys_call

def detect_one_face(rects):
    if len(rects)>1:
        widest_face_num = 0
        curr_max_area = 0
        for j,rect in enumerate(rects):
            (x,y,w,h)=(rect.left(),rect.top(),rect.width(),rect.height())
            if w*h>curr_max_area:
                curr_max_area = w*h
                widest_face_num = j
        return [rects[widest_face_num]]
    else:
        return rects


def trim_overspilled(orig_gray_image,rect):
# function to trim the bounding box so that it doesnt overflow out of the original image
    (x_,y_,w_,h_) = (rect.left(),rect.top(),rect.width(),rect.height())
    rect = dlib.rectangle( max(x_,0), max(y_,0), \
                    min(w_+x_,orig_gray_image.shape[1]-1),\
                    min(h_+y_,orig_gray_image.shape[0]-1))
    return rect
    

def get_crops(frames_folder,crops_folder,folder_no):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    num_images = len(os.listdir(frames_folder))

    metadata = []

    # open lists to collate results and errors/omissions
    result_list = []
    error_list = []

    # Detect face bounding box and extract
    for i in range(1,num_images+1):
        fileName = str(i)+'.jpg'
        print ("Extracting crops for image number ",i,end="\r")
        image = cv2.imread(frames_folder+fileName)
        image = cv2.resize(image, (500, int(image.shape[0]*500/float(image.shape[1]))), cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        # Note if multiple or no faces detected
        if len(rects) > 1 or len(rects) == 0:
            error_list.append([i,len(rects)])
            if len(rects)>1:
                rects = detect_one_face(rects)
            else:
                rects,score,idx = detector.run(gray,1,-1)
                if len(rects) == 0:
                    print("no face detected for ",i)
                    continue
                else:
                    rects = [rects[score.index(max(score))]]
                    (x_,y_,w_,h_) = (rects[0].left(),rects[0].top(),rects[0].width(),rects[0].height())
        
        # the dlib detector might have detected a face bounding box which is not completely contained within the image
        # hence we might need to trim the bounding box to contain it within the image boundary
        rect = trim_overspilled(gray,rects[0])                    
        
        # Face extraction for ith frame
        (x, y, w, h) = (rect.left(),rect.top(),rect.width(),rect.height())
        pred_y,pred_h,pred_x,pred_w = round(y*25/image.shape[0]), round(h*25/image.shape[0]), round(x*25/image.shape[1]), round(w*25/image.shape[1])
        imface = image[y:y+h,x:x+w,:]
        cv2.imwrite(crops_folder+"tabFace/"+fileName,imface)
        metadata.append([i,0,0,pred_x,pred_y,pred_w,pred_h,folder_no,1,0,0])
        result_list.append([i])

        # Eyes extraction for ith frame
        landmarks = predictor(gray, rect)
        landmarks_array = get_landmark_array(landmarks)
        

        crop(crops_folder,i,image,landmarks_array[36:48])

    # Preparing metadata mat file as input to the CSail algorithm
    metadata = np.array(metadata)
    print("array size after extraction ",metadata.shape)
    temp_dict={}
    temp_dict['frameIndex']=metadata[:,0:1].astype(np.int32)
    temp_dict['labelDotXCam']=metadata[:,1:2].astype(np.double)
    temp_dict['labelDotYCam']=metadata[:,2:3].astype(np.double)
    temp_dict['labelFaceGrid']=metadata[:,3:7].astype(np.double)
    temp_dict['labelRecNum']=metadata[:,7:8].astype(np.int16)
    temp_dict['labelTest']=metadata[:,8:9].astype(bool)
    temp_dict['labelTrain']=metadata[:,9:10].astype(bool)
    temp_dict['labelVal']=metadata[:,10:].astype(bool)
    print("dict size ",len(temp_dict))
    sio.savemat('metadata.mat',temp_dict)

    return np.array(error_list),np.array(result_list)

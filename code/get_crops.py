import imutils
from imutils import face_utils
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
            (x,y,w,h)=face_utils.rect_to_bb(rect)
            if w*h>curr_max_area:
                curr_max_area = w*h
                widest_face_num = j
        return [rects[widest_face_num]]
    else:
        return rects


def trim_overspilled(orig_gray_image,rect):
# function to trim the bounding box so that it doesnt overflow out of the original image
    (x_,y_,w_,h_) = face_utils.rect_to_bb(rect)
    rect = dlib.rectangle( max(x_,0), max(y_,0), \
                    min(w_+x_,orig_gray_image.shape[1]-1),\
                    min(h_+y_,orig_gray_image.shape[0]-1))
    return rect
    

def get_crops(frames_folder,crops_folder,folder_no):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    num_images = len(os.listdir(frames_folder))

    csail_metadata = []

    # open lists to collate results and errors/omissions
    result_list = []
    error_list = []

    # Detect face bounding box and extract
    for i in range(1,num_images+1):
        fileName = str(i)+'.jpg'
        print ("Extracting crops for image number ",i,end="\r")
        image = cv2.imread(frames_folder+fileName)
        image = imutils.resize(image, width=500)
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
                    (x_,y_,w_,h_) = face_utils.rect_to_bb(rects[0])
        
        # the dlib detectro might have detected a face bounding box which is not completely contained within the image
        # hence we might need to trim the bounding box to contain it within the image boundary
        rect = trim_overspilled(gray,rects[0])                    
        
        # Face extraction for ith frame
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        pred_y,pred_h,pred_x,pred_w = round(y*25/image.shape[0]), round(h*25/image.shape[0]), round(x*25/image.shape[1]), round(w*25/image.shape[1])
        imface = image[y:y+h,x:x+w,:]
        cv2.imwrite(crops_folder+"tabFace/"+fileName,imface)
        csail_metadata.append([i,0,0,pred_x,pred_y,pred_w,pred_h,folder_no,1,0,0])
        result_list.append([i])
        # str1 = str(i)+",0,0,"+str(pred_x)+","+str(pred_y)+","+str(pred_w)+","+str(pred_h)+ \
        #         ","+str(crops_folder_no)+",1,0,0\n"
        # fil_1.writelines(str1)

        # Eye extraction for ith frame
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        crop(crops_folder,i,image,shape[36:48])

    # Preparing metadata mat file as input to the CSail algorithm
    csail_metadata = np.array(csail_metadata)
    print("array size after extraction ",csail_metadata.shape)
    temp_dict={}
    temp_dict['frameIndex']=csail_metadata[:,0:1].astype(np.int32)
    temp_dict['labelDotXCam']=csail_metadata[:,1:2].astype(np.double)
    temp_dict['labelDotYCam']=csail_metadata[:,2:3].astype(np.double)
    temp_dict['labelFaceGrid']=csail_metadata[:,3:7].astype(np.double)
    temp_dict['labelRecNum']=csail_metadata[:,7:8].astype(np.int16)
    temp_dict['labelTest']=csail_metadata[:,8:9].astype(bool)
    temp_dict['labelTrain']=csail_metadata[:,9:10].astype(bool)
    temp_dict['labelVal']=csail_metadata[:,10:].astype(bool)
    print("dict size ",len(temp_dict))
    sio.savemat('metadata.mat',temp_dict)

    # # Move csail input metadata file to the relevant folder
    # command_string_list = ["mv", csail_metadata, "csail/"]
    # process_descriptor = "moving metadata input file to csail folder"
    # sys_call(command_string_list,process_descriptor)
    # fil_1.close()
    return np.array(error_list),np.array(result_list)

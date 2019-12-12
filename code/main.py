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
    Filename: main.py
'''


import time
#tic = time.perf_counter()
from get_bulk_frames import get_frames
from checkExistence import already_exists
from get_crops import get_crops
import os
from sys_call import sys_call
import numpy as np
from mat_handler import load_mat
from get_external_module import get_module as getmod
# Detached the part for clustering based prediction"
# from kmeans import getClusterPreds


# Provide location of the source videos 
src_path = "../data/videos/"

# location of all data post processing to the 'input' folder 
# (Will have extracted frames, bounding boxes and the metadata file for face grid)
input_folder = "../data/input/"

# location of all WIP data. This will be empty before and after processing
wip_folder = "../WIP/"

# metadata file name
metafile = "metadata.mat"

# summary log file path (Contains count of frames where face detection failed)
summary_file = open("../summary.csv","ab+")
summary_header=b"Child_Id,Total_Frames,Missed_Detection\n"
summary_file.write(summary_header)

#toc = time.perf_counter()
#print("Total import time = ",toc-tic)
#tic = time.perf_counter()

# Extract frames from the respective videos and move them to the input_folder
skip_frame_extraction = False # Toggle this if you have already extracted frames for the vidoe
if not skip_frame_extraction:
    get_frames(src_path,input_folder)

#toc = time.perf_counter()
#print("frames extraction time = ",toc-tic)
#tic = time.perf_counter()

# Extract face and eye crops for each folder numbered on child id
for folder in os.listdir(input_folder):
    # check if metadata already created for this folder. If so, dont create again but just load from the respective file

    if "metadata.mat" not in os.listdir(input_folder+folder+"/"):
        print ("Extracting crops for child ",folder)
        
        # error_list contains the frames where there was a face detection error 'initially'...
        # i.e. where there was no or multiple faces detected. The face might have been finally...
        # detected post adjustment in the sensitivity paramater detector or after correction for multiple faces
        # result_list contains the frame indices for which a face was finally detected
        error_list, result_list = get_crops(input_folder+folder+"/images/",input_folder+folder+"/",folder)

        # Move input metadata file to the relevant folder
        command_string_list = ["mv", "metadata.mat", input_folder+folder+"/"]
        process_descriptor = "moving metadata input file to respective input folder"
        sys_call(command_string_list,process_descriptor)
        
        # Write error log and input frame no. in files to avoid rework in case of a rerun
        if (len(error_list)!=0):
            temp_fil = open(input_folder+folder+"/error_log.csv","wb")
            np.savetxt(temp_fil,error_list)
            temp_fil.close()
        if (len(result_list)!=0):
            temp_fil = open(input_folder+folder+"/input_rows.csv","wb")
            np.savetxt(temp_fil,result_list)
            temp_fil.close()
    else:
        try:
            result_list = np.loadtxt(input_folder+folder+"/input_rows.csv",ndmin=2)
        except:
            pass

        try:
            error_list = np.loadtxt(input_folder+folder+"/error_log.csv")
        except:
            pass

        try:
            if "output_file.csv" in os.listdir(input_folder+folder):
                print("skipping as output.csv exists", folder)
                continue
        except:
            pass


    # load metatdata as numpy array to be written into the output file
    csail_metadata = load_mat(input_folder+folder+"/")

    # # Run the iTracler pytorch code. Before that transfer the images and metadata to a WIP folder
    # # Copy respective metadata file to wip_folder
    # command_string_list = ["cp", input_folder+folder+"/"+metafile,wip_folder]
    # process_descriptor = "copying metadata.mat file to WIP folder"
    # sys_call(command_string_list,process_descriptor)

    print("Final Calculation --- Child ",folder)
    external_module = "main"
    external_module_path = "../csail/main.py"
    external_module = getmod(external_module, external_module_path)
    final_output = external_module.main()
    # final_output = final_output.reshape(-1,1) # undoing squeeze()
    gaze_prediction = np.zeros([final_output.shape[0],1])

    # Customize this part to predict Left or Right from predicted coordinates
    # based on screen dimensions + allowance for error in prediction at edges
    '''
    screen_center = 2
    right_boundary = 20
    left_boundary = -15 
    top_boundary = 3
    bottom_boundary = -15
    gaze_prediction[final_output[:,0:1] > screen_center] = 1
    # weed out exceptions (like gaze out of tablet): shown in the output as label 2
    gaze_prediction[final_output[:,0:1] > right_boundary] = 2
    gaze_prediction[final_output[:,0:1] < left_boundary] = 2
    gaze_prediction[final_output[:,1:2] < bottom_boundary] = 2
    gaze_prediction[final_output[:,1:2] > top_boundary] = 2
    '''

    # Customize this part to weed out exceptions (like very small faces...
    # detected erroneously) shown in the output as label 3
    '''
    gaze_prediction[csail_metadata[:,3:4]*csail_metadata[:,4:5]<=25] = 3
    gaze_prediction = np.array(gaze_prediction)
    '''

    # Writing result to ouptut file for current subject
    final_output = np.concatenate((result_list,csail_metadata,final_output,gaze_prediction),1)
    temp_fil = open(input_folder+folder+"/output_file.csv","wb")
    np.savetxt(temp_fil,final_output)
    temp_fil.close()


    # # Empty out wip_folder
    # command_string_list = ["rm", wip_folder+metafile]
    # process_descriptor = "clearing out WIP folder"
    # sys_call(command_string_list,process_descriptor)

    # Write summary file
    child_id = int(folder) 
    total_frames_extracted = len(final_output)
    total_face_detected = final_output[-1,0]

    # Customize this part to detect count of "out of tablet" gaze based on the screen dimensions
    gaze_out_count = 0
    '''
    for row in final_output:
        if row[7]<left_boundary or row[7]>right_boundary or row[8]<bottom_boundary or row[8]>top_boundary:
            gaze_out_count += 1
    '''
    summary_record = np.array([[child_id,total_frames_extracted,total_frames_extracted-total_face_detected,gaze_out_count]])
    np.savetxt(summary_file, summary_record, delimiter=',', fmt='%05d %d %d %d', comments='')

summary_file.close()

# Detached the part for clustering based prediction"
# getClusterPreds(input_folder)

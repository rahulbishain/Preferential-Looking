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
    Filename: get_bulk_frames.py
'''


from sys_call import sys_call
from checkExistence import already_exists
import os
import re

def get_frames(src_path,input_folder):
    src_vid_list = [f for f in os.listdir(src_path) if f.endswith("looking.mp4")]
    for vid in src_vid_list:
        temp = re.findall("child_(\d+)",vid)
        if temp != []:
            crops_folder_no = temp[0].zfill(5)

            # create folder where all input and output data will be kept
            if not already_exists(input_folder+crops_folder_no):
                # command_string_list = ["mkdir", "../input/"+crops_folder_no]
                # process_descriptor = "creating numbered folder for keeping all data"
                # sys_call(command_string_list,process_descriptor)

                # create folder where all frames will be kept for this source video
                frame_dest_path = input_folder+crops_folder_no+"/images"
                command_string_list = ["mkdir", frame_dest_path]
                process_descriptor = "creating images folder for keeping frames"
                sys_call(command_string_list,process_descriptor)

                # copy source video to the respective numbered input folder
                command_string_list = ["cp", src_path+vid,input_folder+crops_folder_no+"/"+vid]
                process_descriptor = "copying original video to input folder"
                sys_call(command_string_list,process_descriptor)

                # extracting frames
                print ("extracting frames for child ",temp[0])
                command_string_list = ["ffmpeg", "-i", src_path+vid, "-vsync", "2", "-qscale:v", "3", frame_dest_path+"/%d.jpg"]
                process_descriptor = "extracting image frames from video"
                sys_call(command_string_list,process_descriptor)
                print("Total frames extracted: ", len(os.listdir(frame_dest_path)))
    # if (len(result_list)!=len(os.listdir(frame_dest_path))):
    #   print("Image Extraction failed somewhere.. exiting")
    return

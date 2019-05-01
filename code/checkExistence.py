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
    Filename: checkExistence.py
'''


import os
import shutil

def already_exists(to_folder,type="dir",phase_flag=3):
    if type == "dir":
            if os.path.isdir(to_folder):
                if input("Caution! "+ to_folder+" directory already exists. still want to create? (y/n)") != "y":
                    print ("directory not created.")
                    return True
                else:
                    shutil.rmtree(to_folder)
    # In the third phase (phase_flag=3) folders need to be created for iTracker algorithm
    # in the first phase, folder for extracted images and eyes need to be created
            os.makedirs(to_folder,mode=0o744,exist_ok=True)                 #create folder for extracted images or eyes
                                                                            # or numbered folder for iTracker algorithm
    if (phase_flag==3):
            os.makedirs(to_folder+"/tabFace",mode=0o744,exist_ok=True)      #create face folder within it
            os.makedirs(to_folder+"/tabLeftEye",mode=0o744,exist_ok=True)   #create left eye folder within it
            os.makedirs(to_folder+"/tabRightEye",mode=0o744,exist_ok=True)  #create right eye folder within it
    return False

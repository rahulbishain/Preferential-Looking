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
    Filename: boxfun.py
'''


def box(mask,image):
    (im_height,im_width,im_channels) = image.shape 
    # print(mask)
    xmin = mask[:, 0].min() - 10
    xmax = mask[:, 0].max() + 10
    ymin = mask[:,1].min() - 10
    ymax = mask[:,1].max() + 10

    return [max(xmin,0), min(xmax,im_width-1), max(ymin,0), min(ymax,im_height-1)]

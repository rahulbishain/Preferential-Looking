''' 
to generate (Left, Center, Right) Classification Labels for Reference Metadata
Input:
W - Width of device screen
H - Height of device screen
camX - directed distance of camera along width from left
camY - directed distance of camera along height from top
xL - boundary of Left region i.e. x ∈ [0, xL) ⇒ labelDotLCR = 1
xR - boundary of Right region i.e. x ∈ [xR, W) ⇒ labelDotLCR = 3
Note: (camX, camY) have been provided with screen top-left as origin. Also, like MATLAB, it assumes flipped Y-axis
Output:
new field, 'labelDotLCR' in Metadata with values
0 : Gaze Out
1 : Left
2 : Center
3 : Right
 
  Created on: Oct, 2019
      Author: Abhishek Chakraborty
    Filename: process_metadata.py
'''

import os
import scipy.io as sio
import numpy as np
import sys
from generate_LR_labels import read_metadata, save_metadata

def get_labels(metadata, deviceW, deviceH, cameraX, cameraY, xLeft, xRight):
    
    # read X, Y
    xCam = metadata['labelDotXCam']
    yCam = metadata['labelDotYCam']

    # check outside
    outside_left = xCam + cameraX < 0
    outside_right = xCam + cameraX > deviceW
    outside_up = yCam + (deviceH - cameraY) > deviceH
    outside_down = yCam + (deviceH - cameraY) < 0
    outside_all = np.hstack([outside_left, outside_right, outside_up, outside_down])
    outside = np.logical_or.reduce(outside_all.transpose().tolist())

    # check L, R
    assert(xLeft < deviceW and xRight < deviceW)
    isLeft = xCam + cameraX < xLeft
    isRight = xCam + cameraX > xRight

    # assign labels
    labels = np.zeros(xCam.shape, dtype=np.int)
    labels[outside] = 0
    labels[np.logical_and(np.logical_not(outside).reshape(-1,1), isLeft)] = 1
    labels[np.logical_and(np.logical_not(outside).reshape(-1,1), isRight)] = 3
    labels[np.logical_and(  np.logical_not(outside).reshape(-1,1), 
                            np.logical_and( np.logical_not(isLeft).reshape(-1,1), 
                                            np.logical_not(isRight).reshape(-1,1) ))] = 2

    return labels

if __name__ == "__main__":

    # read args
    deviceW = float(sys.argv[1]) # Width
    deviceH = float(sys.argv[2]) # Height
    cameraX = float(sys.argv[3]) # Device Camera location along width from left
    cameraY = float(sys.argv[4]) # Device Camera location along height from top
    xLeft = float(sys.argv[5]) # Left Region boundary
    xRight = float(sys.argv[6]) # Right Region boundary

    # read metadata file
    METADATA_FILE_PATH = '../../reference_metadata.mat'
    metadata = read_metadata(METADATA_FILE_PATH)

    # get labels
    labels = get_labels(metadata, deviceW, deviceH, cameraX, cameraY, xLeft, xRight)

    # write labels
    metadata['labelDotLCR'] = labels
    METADATA_SAVE_PATH = '../../reference_metadata_with_labels.mat'
    save_metadata(METADATA_SAVE_PATH, metadata)
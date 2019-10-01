''' 
to generate Classification Labels for Reference Metadata
Input:
W - Width of device
H - Height of device
camX - location of camera along width from left
camY - location of camera along height from top
Output:
new field, 'labelDotLR' in Metadata with values
0 - Gaze Out
1 - Left
2 - Right
 
  Created on: Sep, 2019
      Author: Abhishek Chakraborty
    Filename: process_metadata.py
'''

import os
import scipy.io as sio
import numpy as np
import sys

def read_metadata(metadataFilePath):
    # metadataFileName = 'reference_metadata.mat'
    # metadataFilePath = os.path.join(datasetPath, metadataFileName)
    metadata = sio.loadmat(metadataFilePath)
    return metadata

def save_metadata(metadataFilePath, content):
    sio.savemat(metadataFilePath, metadata)

def get_labels(metadata, deviceW, deviceH, cameraX, cameraY):
    
    # read X, Y
    xCam = metadata['labelDotXCam']
    yCam = metadata['labelDotYCam']

    # check outside
    outside_left = xCam - cameraX < 0
    outside_right = xCam - cameraX > deviceW
    outside_up = yCam - (deviceH - cameraY) > deviceH
    outside_down = yCam - (deviceH - cameraY) < 0
    outside_all = np.hstack([outside_left, outside_right, outside_up, outside_down])
    outside = np.logical_or.reduce(outside_all.transpose().tolist())

    # check L, R
    isLeft = xCam - cameraX < deviceW/2

    # assign labels
    labels = np.zeros(xCam.shape, dtype=np.int)
    labels[outside] = 0
    labels[np.logical_and(np.logical_not(outside).reshape(-1,1), isLeft)] = 1
    labels[np.logical_and(np.logical_not(outside).reshape(-1,1), np.logical_not(isLeft).reshape(-1,1))] = 2

    return labels

if __name__ == "__main__":

    # read args
    deviceW = float(sys.argv[1]) # Width
    deviceH = float(sys.argv[2]) # Height
    cameraX = float(sys.argv[3]) # Device Camera location along width from left
    cameraY = float(sys.argv[4]) # Device Camera location along height from top

    # read metadata file
    METADATA_FILE_PATH = '../../reference_metadata.mat'
    metadata = read_metadata(METADATA_FILE_PATH)

    # get labels
    labels = get_labels(metadata, deviceW, deviceH, cameraX, cameraY)

    # write labels
    metadata['labelDotLR'] = labels
    METADATA_SAVE_PATH = '../../reference_metadata_with_labels.mat'
    save_metadata(METADATA_SAVE_PATH, metadata)
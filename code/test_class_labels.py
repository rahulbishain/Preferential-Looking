''' 
temporary file containing tests for generate_class_labels.py
 
  Created on: Oct, 2019
      Author: Abhishek Chakraborty
    Filename: process_metadata.py
'''

import numpy as np
from generate_LR_labels import get_labels as get_LR_labels
from generate_LCR_labels import get_labels as get_LCR_labels

def get_test_metadata(xCam, yCam):
    # create Grid
    xCam = np.array(xCam)
    yCam = np.array(yCam)
    x_ind, y_ind = np.meshgrid(xCam, yCam)
    x_ind = x_ind.reshape(-1, 1)
    y_ind = y_ind.reshape(-1, 1)

    # create metadata
    metadata = {}
    metadata['labelDotXCam'] = x_ind
    metadata['labelDotYCam'] = y_ind
    return metadata

# L,R Labels

def get_LR_args():
    return 8, 4, 2, 1

def get_LR_test_metadata():
    xCam = [-3, -1, 1, 3, 7]
    yCam = [2, 0.5, -1, -4]
    metadata = get_test_metadata(xCam, yCam)
    return metadata

def get_LR_test_labels():
    # Label Grid
    labels = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 2, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0]
    ])
    return labels

def test_LR():
    metadata = get_LR_test_metadata()
    deviceW, deviceH, cameraX, cameraY = get_LR_args()
    predicted_labels = get_LR_labels(metadata, deviceW, deviceH, cameraX, cameraY)
    true_labels = get_LR_test_labels()
    matches = predicted_labels == true_labels.reshape(-1, 1)

    print("True Labels:\n", true_labels)
    print("Predicted Labels:\n", predicted_labels.reshape(true_labels.shape))
    print("Accuracy: ", np.sum(matches)/len(true_labels.reshape(-1, 1))*100, "%" )

# L,C,R Labels

def get_LCR_args():
    return 8, 4, 2, 1, 4, 6

def get_LCR_test_metadata():
    xCam = [-3, -1, 1, 3, 5, 7]
    yCam = [2, 0.5, -1, -4]
    metadata = get_test_metadata(xCam, yCam)
    return metadata

def get_LCR_test_labels():
    # Label Grid
    labels = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 2, 3, 0],
        [0, 1, 1, 2, 3, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    return labels

def test_LCR():
    metadata = get_LCR_test_metadata()
    deviceW, deviceH, cameraX, cameraY, xL, xR = get_LCR_args()
    predicted_labels = get_LCR_labels(metadata, deviceW, deviceH, cameraX, cameraY, xL, xR)
    true_labels = get_LCR_test_labels()
    matches = predicted_labels == true_labels.reshape(-1, 1)

    print("True Labels:\n", true_labels)
    print("Predicted Labels:\n", predicted_labels.reshape(true_labels.shape))
    print("Accuracy: ", np.sum(matches)/len(true_labels.reshape(-1, 1))*100, "%" )


if __name__ == "__main__":
    test_LR()
    test_LCR()
''' 
temporary file containing tests for generate_class_labels.py
 
  Created on: Oct, 2019
      Author: Abhishek Chakraborty
    Filename: process_metadata.py
'''

import numpy as np
from generate_class_labels import get_labels

def get_test_args():
    return 8, 4, 2, 1

def get_test_metadata():
    # create Grid
    xCam = np.array([-3, -1, 1, 3, 7])
    yCam = np.array([2, 0.5, -1, -4])
    x_ind, y_ind = np.meshgrid(xCam, yCam)
    x_ind = x_ind.reshape(-1, 1)
    y_ind = y_ind.reshape(-1, 1)

    # create metadata
    metadata = {}
    metadata['labelDotXCam'] = x_ind
    metadata['labelDotYCam'] = y_ind
    return metadata

def get_test_labels():
    # Label Grid
    labels = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 2, 0],
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0]
    ])
    labels = labels.reshape(-1, 1)
    return labels

def test():
    metadata = get_test_metadata()
    deviceW, deviceH, cameraX, cameraY = get_test_args()
    predicted_labels = get_labels(metadata, deviceW, deviceH, cameraX, cameraY)
    true_labels = get_test_labels()
    matches = predicted_labels == true_labels

    print("True Labels: ", true_labels.reshape(-1))
    print("Predicted Labels: ", predicted_labels.reshape(-1))
    print("Accuracy: ", np.sum(matches)/len(true_labels)*100, "%" )

if __name__ == "__main__":
    test()
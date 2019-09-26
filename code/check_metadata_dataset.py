''' 
To verify Preferential Gaze Dataset with its Metadata MATLAB file
 
  Created on: Sep, 2011
      Author: Abhishek Chakraborty
    Filename: process_data.py
'''

import sys
import numpy as np
import process_metadata as pm 
import process_dataset as pd 

if __name__ == "__main__":
    
    datasetPath = sys.argv[1] # 'temp'

    metadata = pm.read_metadata(datasetPath)
    record_list = pm.get_record_list(metadata)

    print("Record: len(json) == len(metadata) AND json XCam == metadata XCam/json YCam == metadata YCam")

    for record in record_list:
        # print("record: ", record)

        # JSON Data
        jsonData = pd.get_record_data(datasetPath, record)
        if not jsonData:
            print(record, ": 'dotInfo.json' not found", )
            continue
        jsonXCam = pd.get_XCam(jsonData)
        jsonYCam = pd.get_YCam(jsonData)
        jsonXCam, jsonYCam = np.array(jsonXCam), np.array(jsonYCam)

        # Dataset
        metadataXCam = pm.get_XCam(metadata, record)
        metadataYCam = pm.get_YCam(metadata, record)

        # print("json XCam.shape : ", jsonXCam.shape[0])
        # print("json YCam.shape : ", jsonYCam.shape[0])
        # print("metadata XCam.shape : ", metadataXCam.shape[0])
        # print("metadata YCam.shape : ", metadataYCam.shape[0])

        assert(jsonXCam.shape[0] == jsonYCam.shape[0])
        assert(metadataXCam.shape[0] == metadataYCam.shape[0])

        # print("jsonXCam: ", jsonXCam)
        # print("metadataXCam: ", metadataXCam)

        if jsonXCam.shape[0] == metadataXCam.shape[0]:
            print(record, ":", jsonXCam.shape[0] == metadataXCam.shape[0], "AND", np.allclose(jsonXCam, metadataXCam, atol=1e-5))
            # print("json XCam == metadata XCam : ", np.allclose(jsonXCam, metadataXCam, atol=1e-5))
            # print("json YCam == metadata YCam : ", np.allclose(jsonYCam, metadataYCam, atol=1e-5))
        else:
            print(record, ":", jsonXCam.shape[0] == metadataXCam.shape[0])
            # print("len(json) == len(metadata) : ", jsonXCam.shape[0] == metadataXCam.shape[0])
            # print("len(json XCam) == len(metadata XCam) : ", jsonXCam.shape[0] == metadataXCam.shape[0])
            # print("len(json YCam) == len(metadata YCam) : ", jsonYCam.shape[0] == metadataYCam.shape[0])
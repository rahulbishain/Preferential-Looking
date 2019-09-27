''' 
To verify Preferential Gaze Dataset with its Metadata MATLAB file
 
  Created on: Sep, 2019
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
    
    print(','.join(['Record', 'len(metadata)', 'len(json)', 'Size Equal', 'Value Equal']))

    for record in record_list:
        # Dataset
        metadataXCam = pm.get_XCam(metadata, record)
        metadataYCam = pm.get_YCam(metadata, record)

        # JSON Data
        jsonData = pd.get_record_data(datasetPath, record)
        if not jsonData:
            csvStr = ','.join(map(str, [record, metadataXCam.shape[0], 'NA', 'NA', 'NA']))
            print(csvStr)
            continue
        jsonXCam = pd.get_XCam(jsonData)
        jsonYCam = pd.get_YCam(jsonData)
        jsonXCam, jsonYCam = np.array(jsonXCam), np.array(jsonYCam)

        assert(jsonXCam.shape[0] == jsonYCam.shape[0])
        assert(metadataXCam.shape[0] == metadataYCam.shape[0])

        # check if equal
        sizeEqual = jsonXCam.shape[0] == metadataXCam.shape[0]
        valueEqual = np.allclose(jsonXCam, metadataXCam, atol=1e-5) if sizeEqual else False
        
        csvStr = ','.join(map(str, [record, metadataXCam.shape[0], jsonXCam.shape[0], sizeEqual, valueEqual]))
        print(csvStr)
''' 
To read and process Preferential Gaze Dataset
 
  Created on: Sep, 2019
      Author: Abhishek Chakraborty
    Filename: process_data.py
'''

import os
import sys
import json

def get_record_data(datasetPath, recordNum):

    # read metadata json
    metadataFileName = 'dotInfo.json'
    recordFolderName = '{:05d}'.format(recordNum)
    try:
        jsonFilePath = os.path.join(datasetPath, recordFolderName, metadataFileName)
        jsonFile = open(jsonFilePath, 'r')
        jsonData = json.load(jsonFile)
    except:
        jsonData = None
    
    return jsonData

def get_XCam(jsonData):
    return jsonData['XCam']

def get_YCam(jsonData):
    return jsonData['XCam']

if __name__ == "__main__":

    datasetPath = sys.argv[1] # 'temp'
    recordNum = sys.argv[2] # 2

    record_data = get_record_data(datasetPath, recordNum)
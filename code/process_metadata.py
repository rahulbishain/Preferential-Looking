''' 
To process Matlab file containing dataset metadata
 
  Created on: Sep, 2019
      Author: Abhishek Chakraborty
    Filename: process_metadata.py
'''

import os
import scipy.io as sio
import numpy as np
import sys

def read_metadata(datasetPath):
    metadataFileName = 'reference_metadata.mat'
    metadataFilePath = os.path.join(datasetPath, metadataFileName)
    metadata = sio.loadmat(metadataFilePath)
    return metadata

def get_XCam(metadata, record_num):
    xCam = metadata['labelDotXCam'][metadata['labelRecNum'] == record_num]
    return xCam

def get_YCam(metadata, record_num):
    yCam = metadata['labelDotXCam'][metadata['labelRecNum'] == record_num]
    return yCam

def get_record_list(metadata):
    record_num_list = np.unique(metadata['labelRecNum'])
    return record_num_list

# def reorganize_metadata(metadata):
    
#     # remove info fields
#     keyList = [ k for k in metadata.keys() if not (k[:2] == '__' and k[-2:] == '__') ] # non private
#     assert(len(set([ len(metadata[k]) for k in keyList ])) == 1) # check if lengths equal

#     # reorganize  
#     metadata_noninfo = dict(zip(keyList, [ metadata[k].tolist() for k in keyList ]))
#     metadata_list = map(dict, zip(*[[(k, v) for v in value] for k, value in metadata_noninfo.items()]))

#     return metadata_list

if __name__ == "__main__":

    # read metadata file
    datasetPath = sys.argv[1] # 'temp'
    recordNum = sys.argv[2] # 2

    metadata = read_metadata(datasetPath) 
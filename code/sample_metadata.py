
import os
import scipy.io as sio
import sys
import numpy as np

SAMPLE_FRACTION = 0.1

# for given key 'binaryLabel', randomly sample 'sampleFraction' fraction of values 
def getBinaryLabelSample(metadataFileDict, binaryLabel, sampleFraction):
    metadataLabel = metadataFileDict[binaryLabel].squeeze()
    labelTrueIndicesList = np.nonzero(metadataLabel == 1)[0] # as tuple
    labelSampledIndices = np.random.choice(labelTrueIndicesList, int(sampleFraction*len(labelTrueIndicesList)), replace=False)
    return labelSampledIndices

# for each label of metadata, get only given indices
def getMetadataForIndices(metadataFileDict, indices):
    subMetadata = {}

    # Non-meta keys
    keyList = getNonMetadataKeys(metadataFileDict)
    for key in keyList:
        subMetadata[key] = metadataFileDict[key][indices,:]
    
    # meta keys
    metaKeyList = list(set(metadataFileDict.keys()).difference(keyList))
    for key in metaKeyList:
        subMetadata[key] = metadataFileDict[key]
    
    return subMetadata

# sample by given fraction
def sampleMetadata(metadataFileDict, sampleFraction=SAMPLE_FRACTION):
    # get indices
    binaryLabels = ['labelTrain', 'labelVal', 'labelTest']
    labelSampledIndices = {}
    for label in binaryLabels:
        labelSampledIndices[label] = getBinaryLabelSample(metadataFileDict, label, sampleFraction)
    sampledIndices = np.hstack(labelSampledIndices.values())

    # sample
    sampledMetadata = getMetadataForIndices(metadataFileDict, sampledIndices)

    return sampledMetadata

def readMetadata(metadataFilePath):
    metadata = sio.loadmat(metadataFilePath)
    return metadata

def writeMetadata(metadataFilePath, metadata):
    sio.savemat(metadataFilePath, metadata)

# get keys with values
def getNonMetadataKeys(metadataFileDict):
    keyList = [key for key in metadataFileDict.keys() if key[:2] != '__'] # non-meta keys
    return keyList

def validateMetadata(metadataFileDict):

    # get keys
    keyList = getNonMetadataKeys(metadataFileDict)

    # checks
    binaryLabels = ['labelTrain', 'labelVal', 'labelTest']
    assert(len(set([len(metadata[key]) for key in keyList])) == 1) # All records of same length
    assert(len(set.difference(set(binaryLabels), set(keyList))) == 0) # 3 Labels exist
    assert(all(map(lambda key: set(metadata[key].squeeze().tolist()) == set([0, 1]), binaryLabels))) # binary labels
    assert(all(map(lambda x: x==1, sum(map(lambda key: metadata[key], binaryLabels))))) # Sum to 1

if __name__ == "__main__":

    metadataFilePath = sys.argv[1] if len(sys.argv) > 1 else '../../temp/revisedMetaFinal.mat'
    saveFilePath = sys.argv[2] if len(sys.argv) > 2 else 'sampledMetadata.mat'
    sampleFraction = float(sys.argv[3]) if len(sys.argv) > 3 else SAMPLE_FRACTION

    # read metadata
    metadata = readMetadata(metadataFilePath)

    # check
    validateMetadata(metadata)

    # sample
    sampledMetadata = sampleMetadata(metadata, sampleFraction)

    # write
    writeMetadata(saveFilePath, sampledMetadata)

import sys
from sample_metadata import readMetadata, writeMetadata, getMetadataForIndices

if __name__ == "__main__":

    labelMetadataFilePath = sys.argv[1] if len(sys.argv) > 1 else '../../Labels/metadata_LR.mat'
    samplingIndicesFilePath = sys.argv[2] if len(sys.argv) > 2 else 'sampledIndices.mat'
    saveFilePath = sys.argv[3] if len(sys.argv) > 3 else 'sampled_metadata_LR.mat'

    # read
    labelMetadata = readMetadata(labelMetadataFilePath)
    samplingIndices = readMetadata(samplingIndicesFilePath)['samplingIndices']

    # sample
    sampledLabels = getMetadataForIndices(labelMetadata, samplingIndices)

    # write
    writeMetadata(saveFilePath, sampledLabels)
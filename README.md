## About ##
Preferential looking tasks routinely deploy gaze tracking for social motivation assessments. Here a user is presented with pairs of adjacent social and non-social images or videos on a screen. The aim is to understand the user's preference for social vs non-social videos. Typically such tasks are carried out in restricted or lab settings with a complicated hardware/software setup which require expert handling and user specific calibration. 
With this approach we aim to build an assessment technique which removes dependence of such tasks on cumbersome setup. It also allows the assessment itself to be carried out on simple tablet devices which do not require expert handling. This is an offline approach and only requires the video of the user performing this task, which is captured by the front camera of the tablet.

## Processing steps ##
1. INPUT - Videos named child_<numeric id>_looking.mp4 to be placed in data/videos/
2. Processing - Run the 'main' python file ET_bulk_pack.py contained in the code folder. To generate clusering based output please run kmeans.py after this step
3. Output - output_file.csv for each subject is placed in data/input/'numeric id'/. Summary file summary.csv for all videos is located at the parent directory level. output_with_clustering.csv is generated if kmeans.py is executed
The individual extracted frames, face and eye crops and metadata .mat file containing binary grid information are also placed here. (A flag for skipping frame extraction is provided in the 'main' file in case frame extraction is not required. The face and eye crop extraction will be skipped in case metadat file is already present in this folder)

## Contents ##
1. Source code folder (code)
2. Retrained iTracker model (checkpoint.pth.tar)
3. WIP folder (utilized during processing. contents deleted after execution)
4. data folder 
4.1 input (extracted frames and metadata placed here)
4.2 videos (the input videos are to be provided here)

### NOTE ###
    There are dummy files in data/input, data/videos and WIP folders. Please remove these files before execution.

## Requirements ##
1. OS: Ubuntu 16 (or any Linux compatible version)
2. Python 3.6
3. Libraries: openCV 3.2, sklearn, dlib
4. Nvidia GPU (8GB or higher)

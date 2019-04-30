Processing steps:
1. INPUT - Videos named child_<numeric id>_looking.mp4 to be placed in data/videos/
2. Processing - Run the 'main' python file ET_bulk_pack.py contained in the code folder. To generate clusering based output please run kmeans.py after this step
3. Output - output_file.csv for each subject is placed in data/input/<numeric id>/. Summary file summary.csv for all videos is located at the parent directory level. output_with_clustering.csv is generated if kmeans.py is executed
The individual extracted frames, face and eye crops and metadata .mat file containing binary grid information are also placed here. (A flag for skipping frame extraction is provided in the 'main' file in case frame extraction is not required. The face and eye crop extraction will be skipped in case metadat file is already present in this folder)

Contents:
1. Source code folder (code)
2. Retrained iTracker model (checkpoint.pth.tar)
3. WIP folder (utilized during processing. contents deleted after execution)
4. data folder 
4.1 input (extracted frames and metadata placed here)
4.2 videos (the input videos are to be provided here)

Requirements:
1. OS: Ubuntu 16 (or any Linux compatible version)
2. Python 3.6
3. Libraries: openCV 3.2, sklearn, dlib
4. Nvidia GPU (8GB or higher)
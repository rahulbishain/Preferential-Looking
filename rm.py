# clean up file to remove previous records

import os

path = "data/input/"

for folder in os.listdir(path):
        os.system("rm -r "+path+folder)

os.system("rm -r summary.csv")

''' 
 Code for analysing Preferential Looking task used for the
 assessment of social motivation using gaze
 Copyright (C) 2018 Rahul Bishain and Sharat Chandran
 
 This file can be redistributed and/or modified under the terms of the 
 GNU General Public License as published by the Free Software 
 Foundation, either version 3 of the License, or (at your option) 
 any later version. Further, the main program utilizes the 
 gaze tracking algorithm iTracker (incuded in the folder named 'csail')
 which is governed by its own licensing terms. Please refer to 
 https://github.com/CSAILVision/GazeCapture for its license agreement
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.


  Created on: Apr, 2018
      Author: Rahul Bishain
    Filename: cropfun.py
'''

import cv2
from boxfun import box

def crop(fil_path,i,image,mask):
    rbox = box(mask[:6],image)
    lbox = box(mask[6:],image)
    lEye_img = image[lbox[2]:lbox[3], lbox[0]:lbox[1]]
    rEye_img = image[rbox[2]:rbox[3], rbox[0]:rbox[1]]
    lEye_img = cv2.resize(lEye_img,(150,100))
    rEye_img = cv2.resize(rEye_img,(150,100))

    cv2.imwrite(fil_path+"tabRightEye/"+str(i)+".jpg",rEye_img)
    cv2.imwrite(fil_path+"tabLeftEye/"+str(i)+".jpg",lEye_img)

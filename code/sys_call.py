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
    Filename: sys_call.py
'''

from subprocess import call
import os

def sys_call(command_string_list,process_descriptor):

    command_log_file = "commandLog.txt"
    command_log = open(command_log_file,"a")

    rc = call(command_string_list,stdout=command_log,stderr=command_log)
    command_log.close()

    if (rc != 0):
        print("Failed while "+process_descriptor+". Return code = ",rc)
    else:
        os.remove(command_log_file)
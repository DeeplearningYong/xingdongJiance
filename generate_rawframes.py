# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2

import copy
import os
import pdb

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")

# min-area to control how many region proposals want, default to be 300; use 10 if want track ball as well 
ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

rawdir= '/Users/davidli/tmp/objExtract/basic-motion-detection/rawFrames'
if not os.path.exists(rawdir):
    os.makedirs(rawdir)

countframe = 0

# loop over the frames of the video
while True:        
	(grabbed, frame) = camera.read()        
        print 'frame counts: '+ str(countframe)
        countframe += 1	# frame start at number 1 
         
	if not grabbed:
		break

	# resize the frame to 640, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width= 640)                

        imgpath = rawdir+'/frame'+str(countframe)+'.jpg'
	cv2.imwrite(imgpath,frame)
		

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

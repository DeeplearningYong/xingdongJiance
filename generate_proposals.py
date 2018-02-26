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

# min-area to control how many region proposals want, default to be 2000=40*50; use 10 if want track ball as well 
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minimum area size")
args = vars(ap.parse_args())

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('output.mp4',fourcc, 10.0, (640,360))


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
anchorFrame = None
countframe = 0 

cropdir= '/Users/davidli/tmp/objExtract/basic-motion-detection/cropFrames'
if not os.path.exists(cropdir):
    os.makedirs(cropdir)

# loop over the frames of the video
while True:
        countBox = 0 # reset for each new frame 	
        
	(grabbed, frame) = camera.read()        
        print 'frame counts: '+ str(countframe)
        countframe += 1	# frame start at number 1 
         
	if not grabbed:
		break

	# resize the frame to 640, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width= 640)                
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if anchorFrame is None:
		anchorFrame = gray
		continue

	# compute the absolute difference between the current frame and anchor frame
	frameDelta = cv2.absdiff(anchorFrame, gray)  # delta frames is empty for the first frame, so delta start at number 2 
        
        # update anchor frame every 10 frames 
        if countframe%10 ==0: 
            anchorFrame = gray

	thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]  # frame difference intensity set to be 10 or larger   

	# dilate the thresholded image to fill in holes, then find contours on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	#(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tmp1, cnts, tmp2= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	#pdb.set_trace()
	backupFrame = copy.copy(frame)  # to write to output 
	for contInd in cnts:
		# if the contour is too small, ignore it
		print 'area size:'+str(cv2.contourArea(contInd))

		if cv2.contourArea(contInd) < args["min_area"]:
			print 'too small, skipped'
			continue		

		# compute the bounding box for the contour, draw it on the frame 
		(x, y, w, h) = cv2.boundingRect(contInd)
                x = int(1*x)
                y = int(1*y)
                w = int(1*w)
                h = int(1*h) 	       
		
          	imgCrop = frame[y:(y+h), x:(x+w)] # crop from origin frame, otherwise get green lines from var:frame		
		cropname = cropdir+'/frame'+str(countframe)+'_crop'+str(countBox)+'.jpg'
		cv2.imwrite(cropname,imgCrop)

		cv2.rectangle(backupFrame, (x, y), (x + w, y + h), (0, 255, 0), 1)		
		countBox += 1 # so count of box also start at number 1 
                print 'countBox: '+str(countBox)		

	out.write(backupFrame)  # write to output video file 
	
        #pdb.set_trace()

        #cv2.imshow("Security Feed", frame)
	#cv2.imshow("Thresh", thresh)
	#cv2.imshow("Frame Delta", frameDelta)
	#key = cv2.waitKey(1) & 0xFF
	#if key == ord("q"):
	#	break

# cleanup the camera and close any open windows
camera.release()
out.release()
cv2.destroyAllWindows()

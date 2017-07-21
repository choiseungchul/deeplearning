# USAGE
# python drone.py --video FlightDemo.mp4

# import the necessary packages
import argparse
import cv2
import numpy as np

# rect = np.zeros((4, 2), dtype="float32")

# load the video
camera = cv2.VideoCapture(0)

T = np.loadtxt('transform_var.conf')

# keep looping
while True:
	# grab the current frame and initialize the status text
	(grabbed, frame) = camera.read()
	status = "No Targets"

	warp = cv2.warpPerspective(frame, T, (550, 550))

	# check to see if we have reached the end of the
	# video

	# show the frame and record if a key is pressed
	cv2.imshow("Frame", frame)
	cv2.imshow('warp', warp)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
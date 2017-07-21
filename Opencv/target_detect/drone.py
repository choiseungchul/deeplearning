# USAGE
# python drone.py --video FlightDemo.mp4

# import the necessary packages
import argparse
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# load the video
camera = cv2.VideoCapture(0)

# keep looping
while True:
	# grab the current frame and initialize the status text
	(grabbed, frame) = camera.read()
	status = "No Targets"

	# check to see if we have reached the end of the
	# video
	if not grabbed:
		break

	# convert the frame to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	edged = cv2.Canny(blurred, 30, 150)

	# find contours in the edge map
	image, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	mask = np.zeros_like(image)

	# loop over the contours
	for c in contours:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)
		area = cv2.contourArea(c)

		if ((len(approx) > 15) & (area > 330)):
			cv2.drawContours(frame, c, -1, (255, 0, 0), 2)
			(x, y, w, h) = cv2.boundingRect(approx)

			cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
			status = "Target(s) Acquired"

			# compute the center of the contour region and draw the
			# crosshairs
			M = cv2.moments(approx)
			(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			(startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
			(startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
			cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
			cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)
			cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
			status = "Target(s) Acquired"


	# draw the status text on the frame
	cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		(0, 0, 255), 2)

	# show the frame and record if a key is pressed
	cv2.imshow("Frame", frame)
	#cv2.imshow("blur", blurred)
	#cv2.imshow("edge", edged)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import os
import requests


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# 비디오 크기
ap.add_argument("-v", "--video", help="path to the video file")
# 감지 움직임 거리
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
# if args.get("video", None) is None:
# 	camera = cv2.VideoCapture(0)
# 	time.sleep(0.25)
#
# # otherwise, we are reading from a video file
# else:

dir_path = os.path.dirname(os.path.realpath(__file__))
#camera = cv2.VideoCapture(dir_path + "/videos/example_02.mp4")
camera = cv2.VideoCapture(0)

# initialize the first frame in the video stream
firstFrame = None


# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()

	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first framedddd
	frameDelta = cv2.absdiff(firstFrame, gray)
	#thresh = cv2.threshold(frameDelta, 35, 255, cv2.THRESH_BINARY_INV)[1]
	thresh = cv2.threshold(frameDelta, 105, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	chkcnt = 0
	# loop over the contours
	for c in contours:
		chkcnt += 1
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)

		print('checked moving object: ', x, y, w , h)

		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"

		# if(chkcnt > 3):
		# 	cv2.imwrite('img_CV2_90.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
		# 	url = 'http://api.vr-storm.net/arc_api/v1/save_target.php'
		# 	files = {'file': open('img_CV2_90.jpg', 'rb')}
		# 	r = requests.post(url, files=files)
		# 	print(r)
		# 	chkcnt = 0
		# 	break


	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()


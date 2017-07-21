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

cv2.namedWindow('variables')

def nothing(b):
	a=0

# create trackbars
cv2.createTrackbar('CannyThreshold1','variables',30,255, nothing)
cv2.createTrackbar('CannyThreshold2','variables',150,255, nothing)
cv2.createTrackbar('ss','variables',144,1000, nothing)
cv2.createTrackbar('solidity','variables',90,100, nothing)
cv2.createTrackbar('dims','variables',25,500, nothing)

rect_size = 550
rect = np.zeros((4, 2), dtype="float32")

# load the video
camera = cv2.VideoCapture(0)

M = None

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
	#edged = cv2.Canny(blurred, 30, 150)
	edged = cv2.Canny(blurred, cv2.getTrackbarPos('CannyThreshold1', 'variables'), cv2.getTrackbarPos('CannyThreshold2', 'variables'))

	# find contours in the edge map
	image, cnts, hi = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	#ratio = frame.shape[0] / 300.0
	orig = frame.copy()

	mask = np.ones(image.shape, dtype="uint8") * 255
	warp = np.ones(image.shape, dtype="uint8") * 255

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		# 0.01 은 선이 부드러워지는 정도
		# approx = cv2.approxPolyDP(c, 0.01 * peri, True)
		approx = cv2.approxPolyDP(c, (cv2.getTrackbarPos('ss', 'variables') / 1000) * peri, True)

		# ensure that the approximated contour is "roughly" rectangular
		if len(approx) >= 4 and len(approx) <= 6:
			screenCnt = approx
			# compute the bounding box of the approximated contour and
			# use the bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			aspectRatio = w / float(h)
			cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)

			# compute the solidity of the original contour
			area = cv2.contourArea(c)
			hullArea = cv2.contourArea(cv2.convexHull(c))
			solidity = area / float(hullArea)

			# compute whether or not the width and height, solidity, and
			# aspect ratio of the contour falls within appropriate bounds

			keepDims = w > cv2.getTrackbarPos('dims', 'variables') and h > cv2.getTrackbarPos('dims', 'variables')

			keepSolidity = solidity > (cv2.getTrackbarPos('solidity', 'variables') / 100)
			# 직사각형의 비율
			keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

			# ensure that the contour passes all our tests
			if keepDims and keepSolidity and keepAspectRatio:
				# draw an outline around the target and update the status
				# text
				cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
				cv2.drawContours(mask, [c], -1, 0, -1)
				status = "Target(s) Acquired"

				# 정사각형으로 변환
				pts = None
				try:
					pts = screenCnt.reshape(4, 2)
				except:
					print("reshape error!")

				# the top-left point has the smallest sum whereas the
				# bottom-right has the largest sum
				if(pts is not None):
					s = pts.sum(axis=1)
					rect[0] = pts[np.argmin(s)]
					rect[2] = pts[np.argmax(s)]

					# compute the difference between the points -- the top-right
					# will have the minumum difference and the bottom-left will
					# have the maximum difference
					diff = np.diff(pts, axis=1)
					rect[1] = pts[np.argmin(diff)]
					rect[3] = pts[np.argmax(diff)]

					# multiply the rectangle by the original ratio
					#rect *= ratio

					# now that we have our rectangle of points, let's compute
					# the width of our new image
					(tl, tr, br, bl) = rect
					widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
					widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

					# ...and now for the height of our new image
					heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
					heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

					# take the maximum of the width and height values to reach
					# our final dimensions
					maxWidth = max(int(widthA), int(widthB))
					maxHeight = max(int(heightA), int(heightB))

					# construct our destination points which will be used to
					# map the screen to a top-down, "birds eye" view

					dst = np.array([
						[0, 0],
						[rect_size, 0],
						[rect_size, rect_size],
						[0, rect_size]], dtype="float32")

					# calculate the perspective transform matrix and warp
					# the perspective to grab the screen
					M = cv2.getPerspectiveTransform(rect, dst)
					warp = cv2.warpPerspective(orig, M, (rect_size, rect_size))

				# compute the center of the contour region and draw the
				# crosshairs
				# M = cv2.moments(approx)
				# (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				# (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
				# (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
				# cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
				# cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)

	# draw the status text on the frame
	cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		(0, 0, 255), 2)

	mask = cv2.bitwise_not(mask)
	frame_crop = cv2.bitwise_and(frame, frame, mask=mask)

	# show the frame and record if a key is pressed
	cv2.imshow("Frame", frame)
	cv2.imshow('edged', edged)
	# cv2.imshow('blurred', blurred)
	cv2.imshow('frame_crop', frame_crop)
	cv2.imshow('warp', warp)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
	if key == ord('s'):
		if M is not None:
			T_var = np.array(M)
			np.savetxt('transform_var.conf', T_var)
			break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
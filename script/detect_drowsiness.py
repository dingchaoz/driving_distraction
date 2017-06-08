# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

"""
	Measure distance between any 2 given points on face
	Default pos1 is nose tip positoin, pos2 is chin position
"""

def measure_distance(pos1=33,pos2=9):

	distance = dist.euclidean(shape[pos1], shape[pos2])
	print (distance)
	return distance

"""
	Measure the ratio of head movement on X and Y directions
"""

def head_x_y_move(nosePos=33,jawPos=9,lbrPos = 20,rbrPos = 25,lfacePos = 3,rfacePos =15):

	nose2jaw = dist.euclidean(shape[nosePos], shape[jawPos])

	foreHeadPos = (shape[lbrPos] + shape[rbrPos]) / 2
	nose2fhead = dist.euclidean(shape[nosePos], foreHeadPos)

	nose2rface = dist.euclidean(shape[nosePos], shape[rfacePos])
	nose2lface = dist.euclidean(shape[nosePos], shape[lfacePos])
	
	xMoveRatio = nose2lface/nose2rface
	yMoveRatio = nose2jaw/nose2fhead

	print(xMoveRatio,yMoveRatio)

	return xMoveRatio,yMoveRatio
	
"""
	Draw any parts of faces given by the starting and ending position numbers
"""

def draw_facepoint(pos1,pos2):
	point = shape[pos1:pos2]
	pointHull = cv2.convexHull(point)
	cv2.drawContours(frame, [pointHull], -1, (0, 255, 0), 1)

def eye_ratio_detect(COUNTER,ALARM_ON):
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		return COUNTER,ALARM_ON

def head_x_turn_detect(COUNTER,ALARM_ON):
		# check to see if the head movement X ratio is below or above the X direction movement
		# threshold, and if so, increment the blink frame counter
		if xMoveRatio < X_LCL or xMoveRatio > X_UCL:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= X_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "HEAD MOVEMENT X ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		return COUNTER,ALARM_ON

def head_y_turn_detect(COUNTER,ALARM_ON):
		# check to see if the head movement Y ratio is below or above the Y direction movement
		# threshold, and if so, increment the blink frame counter
		if yMoveRatio < Y_LCL or yMoveRatio > Y_UCL:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= Y_CONSEC_FRAMES:
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# draw an alarm on the frame
				cv2.putText(frame, "HEAD MOVEMENT Y ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		return COUNTER,ALARM_ON

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# define three constants, one for the head y direction movement Lower Control Limit 
# one for the head y direction movement Upper Control Limit 
# blink and then a second constant for the number of consecutive
# frames the head must be outside of the control limits for to set off the
# alarm

Y_LCL = 0.8
Y_UCL = 1.25
Y_CONSEC_FRAMES = 30

# define three constants, one for the head x direction movement Lower Control Limit 
# one for the head x direction movement Upper Control Limit 
# and then a second constant for the number of consecutive
# frames the head must be outside of the control limits for to set off the
# alarm

X_LCL = 0.5
X_UCL = 2
X_CONSEC_FRAMES = 30

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize the frame counter of head Y direction movement as well as a boolean used to
# indicate if the alarm is going off
HY_COUNTER = 0
HY_ALARM_ON = False

# initialize the frame counter of head X direction movement as well as a boolean used to
# indicate if the alarm is going off
HX_COUNTER = 0
HX_ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(learStart, learEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rearStart, rearEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEar = shape[learStart:learEnd]
		rightEar = shape[rearStart:rearEnd]
		nose = shape[nStart:nEnd]
		mouth = shape[mStart:mEnd]
		jaw = shape[jStart:jEnd]


		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		#headYMove = measure_distance(33,9)
		

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		leftEarHull = cv2.convexHull(leftEar)
		rightEarHull = cv2.convexHull(rightEar)
		noseHull = cv2.convexHull(nose)
		jawHull = cv2.convexHull(jaw)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [leftEarHull], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [rightEarHull], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [nose], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [jaw], -1, (0, 255, 0), 1)
		draw_facepoint(33,35)
		draw_facepoint(8,10)

		COUNTER,ALARM_ON = eye_ratio_detect(COUNTER,ALARM_ON)
		xMoveRatio, yMoveRatio = head_x_y_move()
		HY_COUNTER,HY_ALARM_ON = head_y_turn_detect(HY_COUNTER,HY_ALARM_ON)
		HX_COUNTER,HX_ALARM_ON = head_x_turn_detect(HX_COUNTER,HX_ALARM_ON)
		

		# # check to see if the eye aspect ratio is below the blink
		# # threshold, and if so, increment the blink frame counter
		# if ear < EYE_AR_THRESH:
		# 	COUNTER += 1

		# 	# if the eyes were closed for a sufficient number of
		# 	# then sound the alarm
		# 	if COUNTER >= EYE_AR_CONSEC_FRAMES:
		# 		# if the alarm is not on, turn it on
		# 		if not ALARM_ON:
		# 			ALARM_ON = True

		# 			# check to see if an alarm file was supplied,
		# 			# and if so, start a thread to have the alarm
		# 			# sound played in the background
		# 			if args["alarm"] != "":
		# 				t = Thread(target=sound_alarm,
		# 					args=(args["alarm"],))
		# 				t.deamon = True
		# 				t.start()

		# 		# draw an alarm on the frame
		# 		cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# # otherwise, the eye aspect ratio is not below the blink
		# # threshold, so reset the counter and alarm
		# else:
		# 	COUNTER = 0
		# 	ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
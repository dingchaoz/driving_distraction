# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from constants import *
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
from playsound import playsound
import argparse
import imutils
import time
import dlib
import cv2
import pdb

class Detector(object):
    '''
    A driving distraction detector. For each given frame picture, detect whether the driver in the frame looks distracted/drowsy. 
    '''

    def __init__(self, model_path=None, alarm_path=None):
        
        self.alarm_path = alarm_path
        self.frame      = None
        self.gray_frame = None
        self.rects      = None

        if model_path is None:
            self.face_detector, self.shape_predictor = self.load_face_model('./resource/shape_predictor_68_face_landmarks.dat')
        else:
            self.face_detector, self.shape_predictor = self.load_face_model(model_path) 


    def load_face_model(self, model_path):

        # initialize dlib's face face_detector (HOG-based) and then create
        # the facial landmark shape_predictor
        
        print("[INFO] loading facial landmark shape_predictor...")
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(model_path)

        return face_detector, shape_predictor


    def start_alarm(self):
        # check to see if an alarm file was supplied,
        # and if so, start a thread to have the alarm
        # sound played in the background
        if self.alarm_path is not None:
            t = Thread(target=playsound,
                args=(self.alarm_path,))
            t.deamon = True
            t.start()

        return


    def new_frame(self, frame):
        '''
        read in a new frame and preprocess it
        '''
        self.frame = imutils.resize(frame, width=450)
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        self.rects = self.face_detector(self.gray_frame, 0)


    def eye_aspect_ratio(self, eyel, eyer):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates, averaged between two eyes 
        Al = dist.euclidean(eyel[1], eyel[5])
        Bl = dist.euclidean(eyel[2], eyel[4])

        Ar = dist.euclidean(eyer[1], eyer[5])
        Br = dist.euclidean(eyer[2], eyer[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        Cl = dist.euclidean(eyel[0], eyel[3])
        Cr = dist.euclidean(eyer[0], eyer[3])

        # compute the averaged eye aspect ratio
        ear = 0.5 * ( (Al + Bl) / (2.0 * Cl) + (Ar + Br) / (2.0 * Cr) )

        # return the averaged Eye Aspect Ratio
        return ear


    def measure_distance(self, pos1=33, pos2=9):   
        """
        Measure distance between any 2 given points on face
        Default pos1 is nose tip positoin, pos2 is chin position
        """

        distance = dist.euclidean(shape[pos1], shape[pos2])
        print(distance)
        
        return distance


    def head_x_y_move(self, shape, nosePos=33, jawPos=9, lbrPos = 20, rbrPos = 25, lfacePos = 3, rfacePos =15):
        """
        Measure the ratio of head movement on X and Y directions
        """

        nose2jaw = dist.euclidean(shape[nosePos], shape[jawPos])

        foreHeadPos = (shape[lbrPos] + shape[rbrPos]) / 2
        nose2fhead = dist.euclidean(shape[nosePos], foreHeadPos)

        nose2rface = dist.euclidean(shape[nosePos], shape[rfacePos])
        nose2lface = dist.euclidean(shape[nosePos], shape[lfacePos])
        
        xMoveRatio = nose2lface / nose2rface
        yMoveRatio = nose2jaw / nose2fhead

        print('xMoveRatio: {}; yMoveRatio: {}'.format(xMoveRatio, yMoveRatio))

        return xMoveRatio, yMoveRatio
    

    def draw_facepoint(self, shape, pos1, pos2):
        """
        Draw any parts of faces on the frame, given the starting and ending position numbers
        """
        point = shape[pos1:pos2]
        pointHull = cv2.convexHull(point)
        
        cv2.drawContours(self.frame, [pointHull], -1, (0, 255, 0), 1)


    def eye_ratio_detect(self, ear, shape, COUNTER, ALARM_ON):
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the drowsiness frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                    self.start_alarm()

                # draw an alarm on the frame
                cv2.putText(self.frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER  = 0
            ALARM_ON = False

        return COUNTER, ALARM_ON


    def head_x_turn_detect(self, xMoveRatio, shape, COUNTER, ALARM_ON):
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
                        self.start_alarm()

                    # draw an alarm on the frame
                    cv2.putText(self.frame, "HEAD MOVEMENT X ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER  = 0
                ALARM_ON = False

            return COUNTER, ALARM_ON


    def head_y_turn_detect(self, yMoveRatio, shape, COUNTER, ALARM_ON):
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
                        self.start_alarm()

                    # draw an alarm on the frame
                    cv2.putText(self.frame, "HEAD MOVEMENT Y ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER  = 0
                ALARM_ON = False

            return COUNTER, ALARM_ON


    def analyze_frame(self, COUNTER, ALARM_ON, HY_COUNTER, HY_ALARM_ON, HX_COUNTER, HX_ALARM_ON):
        '''
        process the current frame, detect its face components and decide if drowsiness/distraction is present
        '''
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        for rect in self.rects:    
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = face_utils.shape_to_np( self.shape_predictor(self.gray_frame, rect) )

            leftEye     = shape[lStart:lEnd]
            rightEye    = shape[rStart:rEnd]
            leftEar     = shape[learStart:learEnd]
            rightEar    = shape[rearStart:rearEnd]
            nose        = shape[nStart:nEnd]
            mouth       = shape[mStart:mEnd]
            jaw         = shape[jStart:jEnd]

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull  = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            leftEarHull  = cv2.convexHull(leftEar)
            rightEarHull = cv2.convexHull(rightEar)
            noseHull     = cv2.convexHull(nose)
            jawHull      = cv2.convexHull(jaw)
            mouthHull    = cv2.convexHull(mouth)
            cv2.drawContours(dt.frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(dt.frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [leftEarHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEarHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [nose], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [jaw], -1, (0, 255, 0), 1)
            dt.draw_facepoint(shape, 33, 35)
            dt.draw_facepoint(shape, 8, 10)


            ear = dt.eye_aspect_ratio(leftEye, rightEye)
            xMoveRatio, yMoveRatio = dt.head_x_y_move(shape)
            #headYMove = measure_distance(33,9)
         
            COUNTER, ALARM_ON       = dt.eye_ratio_detect(ear, shape, COUNTER, ALARM_ON)            
            HY_COUNTER, HY_ALARM_ON = dt.head_y_turn_detect(yMoveRatio, shape, HY_COUNTER, HY_ALARM_ON)
            HX_COUNTER, HX_ALARM_ON = dt.head_x_turn_detect(xMoveRatio, shape, HX_COUNTER, HX_ALARM_ON)
            

            cv2.putText(dt.frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(dt.frame, "X: {:.2f}".format(xMoveRatio), (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)                  
            
            cv2.putText(dt.frame, "Y: {:.2f}".format(yMoveRatio), (300, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return COUNTER, ALARM_ON, HY_COUNTER, HY_ALARM_ON, HX_COUNTER, HX_ALARM_ON



if __name__ == '__main__':

    # start the video stream thread
    model_path = "./resource/shape_predictor_68_face_landmarks.dat"
    alarm_path = "./resource/alarm.wav"
    dt = Detector(model_path, alarm_path)
    webcam = 0

    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=webcam).start()
    time.sleep(1.0)

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
    
    # loop over frames from the video stream
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels)
        dt.new_frame(vs.read())

        # draw facial landmarks?
     
        COUNTER, ALARM_ON, HY_COUNTER, HY_ALARM_ON, HX_COUNTER, HX_ALARM_ON = dt.analyze_frame( \
                                                                                COUNTER, ALARM_ON, \
                                                                                HY_COUNTER, HY_ALARM_ON, \
                                                                                HX_COUNTER, HX_ALARM_ON)  

        # show the frame
        cv2.imshow("Frame", dt.frame)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


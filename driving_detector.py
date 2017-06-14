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
import speech_recognition as sr
import argparse
import imutils
import time
import dlib
import cv2
import pdb
import os

class Detector(object):
    '''
    A driving distraction detector. For each given frame picture, detect whether the driver in the frame looks distracted/drowsy. 
    '''

    def __init__(self, model_path=None, alarm_path=None):
        
        self.alarm_path = alarm_path
        self.frame      = None
        self.gray_frame = None
        self.rects      = None

        self.EAR_COUNTER, self.HY_COUNTER, self.HX_COUNTER,self.MAR_COUNTER  = (0, 0, 0,0)
        self.EAR_ALARM_ON, self.HY_ALARM_ON, self.HX_ALARM_ON ,self.MAR_ALARM_ON  = (False, False, False,False)

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


    def start_alarm(self,message):
        # check to see if an alarm file was supplied,
        # and if so, start a thread to have the alarm
        # sound played in the background

        if self.alarm_path is not None:
            t = Thread(target=os.system,
                args=('say -v Victoria ' + message,))
            t.deamon = True
            t.start()

        return


    def new_frame(self, frame, width=450):
        '''
        read in a new frame and preprocess it
        '''
        self.frame = imutils.resize(frame, width=width)
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        self.rects = self.face_detector(self.gray_frame, 0)


    def read(self):
        '''
        be compatible with cv2.VideoCapture()
        '''
        return self.frame


    def get_iris(self,eye):

        iris_pos = (eye[0]+eye[3])/2
        iris_pos = [int(x) for x in iris_pos]
        iris_pos = tuple(iris_pos)
        
        return iris_pos


    def xy_diff(self,xMoveRatio,yMoveRatio,X_LCL,X_UCL,Y_LCL,Y_UCL):

        cal_xRatio = (X_LCL + X_UCL)/2
        cal_yRatio = (Y_LCL + Y_UCL)/2
        xDiff = xMoveRatio - cal_xRatio
        yDiff = yMoveRatio - cal_yRatio

        #print(cal_xRatio,cal_yRatio,xDiff,yDiff)

        return xDiff,yDiff

    def draw_sight(self,iris_pos,xDiff,yDiff):
 
        # if xMoveRatio < 1:
        #     xMoveRatio = -1/xMoveRatio


        # if yMoveRatio < 1:
        #     yMoveRatio /=-1

        sight_x = int(iris_pos[0] + xDiff*50)
        sight_y = int(iris_pos[1] + yDiff*30)

        sight_pos = (sight_x,sight_y)

        return sight_pos


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

    def mouth_aspect_ratio(self,shape):
        # compute the width and height of mouth
        width = dist.euclidean(shape[49], shape[55])
        height = dist.euclidean(shape[52], shape[58])
        mar = width/height

        # return the eye aspect ratio
        return mar


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

        #print('xMoveRatio: {}; yMoveRatio: {}'.format(xMoveRatio, yMoveRatio))

        return xMoveRatio, yMoveRatio


    def connect_facepoint(self,shape,pos1,pos2,rgb = (255,0,0)):
        for pos in range(pos1,pos2):

            cv2.line(self.frame,tuple(shape[pos]),tuple(shape[pos+1]),rgb,1)
    

    def draw_facepoint(self, shape, pos1, pos2):
        """
        Draw any parts of faces on the frame, given the starting and ending position numbers
        """
        point = shape[pos1:pos2]
        pointHull = cv2.convexHull(point)
        
        cv2.drawContours(self.frame, [pointHull], -1, (0, 255, 0), 1)


    def eye_ratio_detect(self, ear, shape):
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the drowsiness frame counter
        if ear < EYE_AR_THRESH:
            self.EAR_COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if self.EAR_COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not self.EAR_ALARM_ON:
                    self.EAR_ALARM_ON = True
                    self.start_alarm('Hey Wake up')

                # draw an alarm on the frame
                cv2.putText(self.frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            self.EAR_COUNTER  = 0
            self.EAR_ALARM_ON = False

    def mouth_ratio_detect(self,mar,shape):
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if mar < MOUTH_AR_THRESH:
            self.MAR_COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if self.MAR_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not self.MAR_ALARM_ON:
                    self.MAR_ALARM_ON = True
                    self.start_alarm('Your mouth can fit an elepant')

                # draw an alarm on the frame
                cv2.putText(self.frame, "YARN ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            self.MAR_COUNTER = 0
            self.MAR_ALARM_ON = False



    def head_x_turn_detect(self, xMoveRatio, X_LCL,X_UCL):
        # check to see if the head movement X ratio is below or above the X direction movement
        # threshold, and if so, increment the blink frame counter
        if xMoveRatio < X_LCL or xMoveRatio > X_UCL:
            self.HX_COUNTER += 1
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if self.HX_COUNTER >= X_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not self.HX_ALARM_ON:
                    self.HX_ALARM_ON = True
                    self.start_alarm('Watch ahead please')

                # draw an alarm on the frame
                cv2.putText(self.frame, "HEAD MOVEMENT X ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            self.HX_COUNTER  = 0
            self.HX_ALARM_ON = False


    def head_y_turn_detect(self, yMoveRatio, Y_LCL,Y_UCL):
        # check to see if the head movement Y ratio is below or above the Y direction movement
        # threshold, and if so, increment the blink frame counter
        if yMoveRatio < Y_LCL or yMoveRatio > Y_UCL:
            self.HY_COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if self.HY_COUNTER >= Y_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not self.HY_ALARM_ON:
                    self.HY_ALARM_ON = True
                    if yMoveRatio > Y_UCL:
                        self.start_alarm('Are you staring at a flying pig ?')
                    if yMoveRatio < Y_LCL:
                        self.start_alarm(' Did you notice an hundred dollar bill on the mat ?')


                # draw an alarm on the frame
                cv2.putText(self.frame, "HEAD MOVEMENT Y ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            self.HY_COUNTER  = 0
            self.HY_ALARM_ON = False


    def calibrate(self,xMoveRatio,yMoveRatio):      
        global FIRST5_FRAME,X_UCL,X_LCL,Y_UCL,Y_LCL,X_UCL_ARR,X_LCL_ARR,Y_LCL_ARR,Y_UCL_ARR
        
        if FIRST5_FRAME <= 5:

                x_lcl = xMoveRatio*0.5
                x_ucl = xMoveRatio*2

                y_lcl = yMoveRatio*.8
                y_ucl = yMoveRatio*1.25

                X_LCL_ARR.append(x_lcl)
                X_UCL_ARR.append(x_ucl)
                Y_LCL_ARR.append(y_lcl)
                Y_UCL_ARR.append(y_ucl)

                X_LCL = np.mean(X_LCL_ARR)
                X_UCL = np.mean(X_UCL_ARR)
                Y_LCL = np.mean(Y_LCL_ARR)
                Y_UCL = np.mean(Y_UCL_ARR)

                #print('CALIBRATION',X_UCL_ARR)

                FIRST5_FRAME+=1
                #print (FIRST5_FRAME)


    def wuguan_outline(self,shape,leftEyeHull,rightEyeHull):
        # Draw eye brow outlines
        self.connect_facepoint(shape,17,21,EYE_RGB)
        self.connect_facepoint(shape,22,26,EYE_RGB)
        # Draw mouth outline
        self.connect_facepoint(shape,48,60,EYE_RGB)
        # Draw face outline
        self.connect_facepoint(shape,0,16,EYE_RGB)
        # Draw nose outline
        self.connect_facepoint(shape,27,30,EYE_RGB)
        self.connect_facepoint(shape,31,35,EYE_RGB)
        # Draw eye outline
        cv2.drawContours(self.frame, [leftEyeHull], -1, EYE_RGB, 1)
        cv2.drawContours(self.frame, [rightEyeHull], -1, EYE_RGB, 1)


    def eye_areamtrix(self,shape):

        cv2.circle(self.frame,tuple(shape[36]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[19]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[39]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[42]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[24]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[45]),2,EYE_RGB,1)

        cv2.line(self.frame,tuple(shape[36]),tuple(shape[19]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[36]),tuple(shape[39]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[19]),tuple(shape[39]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[39]),tuple(shape[42]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[42]),tuple(shape[24]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[19]),tuple(shape[24]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[24]),tuple(shape[45]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[42]),tuple(shape[45]),EYE_RGB,1)


    def eye2nose_areamtrix(self,shape):

        cv2.circle(self.frame,tuple(shape[33]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[31]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[35]),2,EYE_RGB,1)

        cv2.line(self.frame,tuple(shape[36]),tuple(shape[33]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[45]),tuple(shape[33]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[39]),tuple(shape[33]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[42]),tuple(shape[33]),EYE_RGB,1)
        # cv2.line(self.frame,tuple(shape[31]),tuple(shape[33]),EYE_RGB,1)
        # cv2.line(self.frame,tuple(shape[33]),tuple(shape[35]),EYE_RGB,1)


    def mouth_areamtrix(self,shape):

        cv2.circle(self.frame,tuple(shape[48]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[54]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[8]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[4]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[12]),2,EYE_RGB,1)

        cv2.line(self.frame,tuple(shape[48]),tuple(shape[33]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[54]),tuple(shape[33]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[48]),tuple(shape[8]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[54]),tuple(shape[8]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[4]),tuple(shape[8]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[4]),tuple(shape[48]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[12]),tuple(shape[8]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[12]),tuple(shape[54]),EYE_RGB,1)


    def cheek_areamtrix(self,shape):

        cv2.circle(self.frame,tuple(shape[0]),2,EYE_RGB,1)
        cv2.circle(self.frame,tuple(shape[16]),2,EYE_RGB,1)
        # cv2.circle(self.frame,tuple(shape[8]),2,EYE_RGB,1)
        # cv2.circle(self.frame,tuple(shape[4]),2,EYE_RGB,1)
        # cv2.circle(self.frame,tuple(shape[12]),2,EYE_RGB,1)


        cv2.line(self.frame,tuple(shape[16]),tuple(shape[12]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[16]),tuple(shape[24]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[16]),tuple(shape[45]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[0]),tuple(shape[4]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[0]),tuple(shape[19]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[0]),tuple(shape[36]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[4]),tuple(shape[31]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[4]),tuple(shape[36]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[12]),tuple(shape[35]),EYE_RGB,1)
        cv2.line(self.frame,tuple(shape[12]),tuple(shape[45]),EYE_RGB,1)


    def face_match(self):
        face_locations = face_recognition.face_locations(self.frame)
        face_encodings = face_recognition.face_encodings(self.frame, face_locations)

        for i in range(len(face_locations)):

            #print (len(face_locations))

            face_encoding = face_encodings[i]
            face_location = face_locations[i]
             # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding,tolerance = 0.6)
            

            indices_match = np.where(match)[0]
            print ('indice_match',indices_match)
            if indices_match >= 0:
                
                print (know_faces_names[indices_match[0]])
            else:
                 print ('unknown face')




    def analyze_frame(self):
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

            self.face_match()

            # Get the face rectangular 
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            leftEye     = shape[lStart:lEnd]
            rightEye    = shape[rStart:rEnd]
            leftEar     = shape[learStart:learEnd]
            rightEar    = shape[rearStart:rearEnd]
            nose        = shape[nStart:nEnd]
            mouth       = shape[mStart:mEnd]
            jaw         = shape[jStart:jEnd]
            lIris = self.get_iris(leftEye)
            rIris = self.get_iris(rightEye)

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull  = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            leftEarHull  = cv2.convexHull(leftEar)
            rightEarHull = cv2.convexHull(rightEar)
            noseHull     = cv2.convexHull(nose)
            jawHull      = cv2.convexHull(jaw)
            mouthHull    = cv2.convexHull(mouth)

            cv2.rectangle(self.frame, (x, y), (x + w, y + h), EYE_RGB, 1)
            ##cv2.drawContours(self.frame, [nose], -1, (179,66,244), 1)
            #cv2.drawContours(self.frame, [mouth], -1, (179,66,244), 1)
            #cv2.drawContours(self.frame, [leftEarHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(self.frame, [rightEarHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(self.frame, [jaw], -1, (0, 255, 0), 1)

            #self.wuguan_outline(shape,leftEyeHull,rightEyeHull)
            
            

            ear = self.eye_aspect_ratio(leftEye, rightEye)
            mar = self.mouth_aspect_ratio(shape)
            xMoveRatio, yMoveRatio = self.head_x_y_move(shape)
            self.calibrate(xMoveRatio,yMoveRatio)

            self.eye_ratio_detect(ear, shape)
            self.mouth_ratio_detect(mar,shape)            
            self.head_y_turn_detect(yMoveRatio, Y_LCL,Y_UCL)
            self.head_x_turn_detect(xMoveRatio, X_LCL,X_UCL)
            xDiff,yDiff = self.xy_diff(xMoveRatio,yMoveRatio,X_LCL,X_UCL,Y_LCL,Y_UCL)

            self.eye_areamtrix(shape)
            self.eye2nose_areamtrix(shape)
            self.mouth_areamtrix(shape)
            self.cheek_areamtrix(shape)

            cv2.line(self.frame,lIris,self.draw_sight(lIris,xDiff,yDiff),SIGHT_RGB,1)
            cv2.line(self.frame,rIris,self.draw_sight(rIris,xDiff,yDiff),SIGHT_RGB,1)
            
            # cv2.putText(self.frame, "EAR: {:.2f}".format(ear), (300, 120),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # cv2.putText(self.frame, "MAR: {:.2f}".format(mar), (300, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # cv2.putText(self.frame, "X: {:.2f}".format(xMoveRatio), (300, 60),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)                  
            
            # cv2.putText(self.frame, "Y: {:.2f}".format(yMoveRatio), (300, 90),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try: 
            return (ear, self.EAR_ALARM_ON, \
                    mar, self.MAR_ALARM_ON,\
                    yMoveRatio, self.HY_ALARM_ON, \
                    xMoveRatio, self.HX_ALARM_ON)
        except:
            return (None, self.EAR_ALARM_ON, \
                    None, self.MAR_ALARM_ON,\
                    None, self.HY_ALARM_ON, \
                    None, self.HX_ALARM_ON)



if __name__ == '__main__':

    # start the video stream thread
    model_path = "./resource/shape_predictor_68_face_landmarks.dat"
    alarm_path = "./resource/alarm.wav"
    dt = Detector(model_path, alarm_path)
    webcam = 0

    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=webcam).start()
    time.sleep(1.0)
    
    # loop over frames from the video stream
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels)
        dt.new_frame(vs.read())
     
        ear, EAR_ALARM_ON, mar, MAR_ALARM_ON, yMoveRatio, HY_ALARM_ON, xMoveRatio, HX_ALARM_ON = dt.analyze_frame()

        # show the frame
        cv2.imshow("Frame", dt.frame)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


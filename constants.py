
from imutils import face_utils
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

MOUTH_AR_THRESH = 1.0
MOUTH_AR_CONSEC_FRAMES = 48

# define three constants, one for the head y direction movement Lower Control Limit 
# one for the head y direction movement Upper Control Limit 
# blink and then a second constant for the number of consecutive
# frames the head must be outside of the control limits for to set off the
# alarm
Y_LCL = None
Y_UCL = None
Y_LCL_ARR = []
Y_UCL_ARR = []
Y_CONSEC_FRAMES = 30

# define three constants, one for the head x direction movement Lower Control Limit 
# one for the head x direction movement Upper Control Limit 
# and then a second constant for the number of consecutive
# frames the head must be outside of the control limits for to set off the
# alarm
X_LCL = None
X_UCL = None
X_LCL_ARR = []
X_UCL_ARR = []
X_CONSEC_FRAMES = 30


# count the first 5 frame
FIRST5_FRAME = 0

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(learStart, learEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rearStart, rearEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

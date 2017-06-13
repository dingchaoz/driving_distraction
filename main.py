from driving_detector import Detector
from imutils.video import VideoStream
from GUI import GUI
import tkinter as tk
import time

model_path = "./resource/shape_predictor_68_face_landmarks.dat"
alarm_path = "./resource/alarm.wav"
webcam = 0

dt = Detector(model_path, alarm_path)

print("[INFO] starting video stream thread...")
#vs = cv2.VideoCapture(webcam)
vs = VideoStream(src=webcam).start()
time.sleep(1.0)

root = tk.Tk()
root.wm_title("Driving Monitor")
myapp = GUI(root, dt, vs)

tk.mainloop()


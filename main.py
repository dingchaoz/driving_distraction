from detect_drowsiness_rewrite import Detector
from GUI import GUI
import tkinter as tk

model_path = "../resource/shape_predictor_68_face_landmarks.dat"
alarm_path = "../resource/alarm.wav"
webcam = 0

dt = Detector(model_path, alarm_path)
vs = cv2.VideoCapture(webcam)

root = tk.Tk()
root.wm_title("Driving Monitor")
myapp = GUI(root, vs)

tk.mainloop()


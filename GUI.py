from PIL import Image
from PIL import ImageTk
import speech_recognition as sr
import tkinter as tk
import threading
import logging
import imutils
import cv2
import os

class GUI(object):

    def __init__(self, root, dt, vs, width=1000):
        '''
        Initialize the basic frames under the root window 
        '''
        self._root = root
        self._vs = vs
        self._dt = dt
        self._start = False
        self._frame_width = width
        
        self.logoFrame = tk.Frame(self._root)
        self.logoFrame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))

        self.displayFrame = tk.Frame(self._root)
        self.displayFrame.grid(row = 1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.logger = logging.getLogger(__name__)

        self._root.bind('<Escape>', lambda e: self.quit())
        self.createLogoFrame()
        self.createDisplayFrame()
        self.show_frame()


    def createLogoFrame(self):
        '''
        Display the statefarm logo on top
        '''
        logo = cv2.imread("./resource/logo.jpg")
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
        logo = imutils.resize(logo, width=800)

        logo = Image.fromarray(logo)
        logo = ImageTk.PhotoImage(logo)

        self.logo_label = tk.Label(image=logo, bg='#{0:x}{1:x}{2:x}'.format(211,19,30))
        self.logo_label.image = logo
        self.logo_label.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W, pady=5)


    def createVideoLabel(self):
        '''
        Create the video display
        '''
        self.video_label = tk.Label(self.displayFrame)
        self.video_label.grid(row=0, column=0, rowspan=8, sticky=tk.N+tk.S+tk.E+tk.W, padx=10, pady=10)


    def show_frame(self):
        '''
        Display the video frame by frame
        '''
        if self._start == True:
            #_, frame = self._vs.read()
            #frame = cv2.flip(frame, 1)
            self._dt.new_frame(self._vs.read(), width=int(self._frame_width*0.6))
            status = self._dt.analyze_frame()
            frame = self._dt.read()
            
        else:
            frame = cv2.imread('./resource/idle.png')
            status = (None, False, None, False, None, False)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = imutils.resize(cv2image, width=self._frame_width)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.displayFrame.update()

        # update status in GUI
        

        self.video_label.after(10, self.show_frame) 


    def createDisplayFrame(self):
        '''
        Create the status display frame
        '''
        # Video Display
        self.createVideoLabel()

        # Status Textbox
        self.status_label = tk.Label(self.displayFrame, text='Driver Status', font=('-weight bold'))
        self.status_label.grid(row=0, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))

        self.status_msg = tk.StringVar()
        self.status_msg.set('\nNORMAL!\n')
        self.status_box = tk.Message(self.displayFrame, textvariable=self.status_msg, relief=tk.GROOVE, justify=tk.CENTER, padx=5,
                             fg='green', font=('Helvetica', 24, 'bold'))
        self.status_box.grid(row=1, column=1, rowspan=2, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10), pady=(0, 10))

        # EAR Textbox
        self.EAR_label = tk.Label(self.displayFrame, text='Eye Aspect Ratio', font=('-weight bold'))
        self.EAR_label.grid(row=3, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))

        self.EAR_msg = tk.StringVar()
        self.EAR_msg.set('1.0')
        self.EAR_box = tk.Message(self.displayFrame, textvariable=self.EAR_msg, relief=tk.GROOVE, justify=tk.CENTER, padx=5)
        self.EAR_box.grid(row=4, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))        

        # Head Position
        self.head_label = tk.Label(self.displayFrame, text='Head Position', font=('-weight bold'))
        self.head_label.grid(row=5, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))        

        self.head_msg = tk.StringVar()
        self.head_msg.set('V Ratio: \nH Ratio: ')
        self.head_box = tk.Message(self.displayFrame, textvariable=self.head_msg, relief=tk.GROOVE, justify=tk.CENTER, padx=5)        
        self.head_box.grid(row=6, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10)) 

        # Start Button
        self.start_btn = tk.Button(self.displayFrame, text='START', command=self.onStart, font=('-weight bold'))
        self.start_btn.grid(row=7, column=1, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 0))

        # Stop Button
        self.stop_btn = tk.Button(self.displayFrame, text='STOP', command=self.onStop, font=('-weight bold'))
        self.stop_btn.grid(row=7, column=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(0, 10))


    def voice_command(self):
        # for testing purposes, we're just using the default API key
        GOOGLE_SPEECH_RECOGNITION_API_KEY = None

        r = sr.Recognizer()
        with sr.Microphone() as source:

            while True:
                self.logger.debug("Awaiting user input.")
                audio = r.listen(source)

                self.logger.debug("Attempting to transcribe user input.")

                try:
                    result = r.recognize_google(audio,
                                                key=GOOGLE_SPEECH_RECOGNITION_API_KEY)

                    if result == 'start':
                        self.onStart()

                    elif result == 'stop':
                        self.onStop()

                except sr.UnknownValueError:
                    self.logger.debug("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    self.logger.warn("Could not request results from Google Speech Recognition service: %s", e)
                except Exception as e:
                    self.logger.error("Could not process text: %s", e)





    def onStop(self):
        '''
        Stop video streaming
        '''
        self._start = False
        print('Video Streaming Stopped...')


    def onStart(self):
        '''
        Start video streaming
        '''
        self._start = True
        print('Video Streaming Started...')


    def update_status(self, type=None):
        '''
        Update driver status in the display frame
        type: 'drowsiness', 'distraction', None
        '''
        if type == 'drowsiness':
            self.status_msg.set("Drowsiness ALERT!")
            self.status_label.config(fg='red2')
        elif type == 'distraction':
            self.status_msg.set('Distraction ALERT!')
            self.status_label.config(fg='dark orange')
        else:
            self.status_msg.set('Normal')
            self.status_label.config(fg='green3')


    def quit(self):
        '''
        Clean up the camera and TK windows
        '''
        print("[INFO] Closing the application...")
        #self._vs.release()
        self._vs.stop()
        self._root.quit()




if __name__ == '__main__':

    root = tk.Tk()
    root.wm_title("Driving Monitor")

    #vs = cv2.VideoCapture(0)

    myapp = GUI(root, vs)

    #myapp.voice_command()

    tk.mainloop()



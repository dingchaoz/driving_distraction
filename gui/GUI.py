from PIL import Image
from PIL import ImageTk
import tkinter as tk
import threading
import imutils
import cv2
import os

class GUI(object):

    def __init__(self, root, vs=None):
        '''
        Initialize the basic frames under the root window 
        '''
        self._root = root
        self._vs = vs
        self._start = False

        self.logoFrame = tk.Frame(self._root)
        self.logoFrame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))

        self.displayFrame = tk.Frame(self._root)
        self.displayFrame.grid(row = 1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        self._root.bind('<Escape>', lambda e: self.quit())
        self.createLogoFrame()
        self.createDisplayFrame()
        self.show_frame()


    def createLogoFrame(self):
        '''
        Display the statefarm logo on top
        '''
        logo = cv2.imread("./logo1.jpg")
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
        logo = imutils.resize(logo, width=1000)

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
            _, frame = self._vs.read()
            frame = cv2.flip(frame, 1)
        else:
            frame = cv2.imread('./idle.png')

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = imutils.resize(cv2image, width=1000)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.displayFrame.update()

        self.video_label.after(10, self.show_frame) 


    def createDisplayFrame(self):
        '''
        Create the status display frame
        '''
        # Video Display
        self.createVideoLabel()

        # Status Textbox
        self.status_label = tk.Label(self.displayFrame, text='Driver Status')
        self.status_label.grid(row=0, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))

        self.status_msg = tk.StringVar()
        self.status_msg.set('\nNORMAL!\n')
        self.status_box = tk.Message(self.displayFrame, textvariable=self.status_msg, relief=tk.GROOVE, justify=tk.CENTER, padx=5,
                             fg='green', font=('Helvetica', 24, 'bold'))
        self.status_box.grid(row=1, column=1, rowspan=2, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10), pady=(0, 10))

        # EAR Textbox
        self.EAR_label = tk.Label(self.displayFrame, text='Eye Aspect Ratio')
        self.EAR_label.grid(row=3, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))

        self.EAR_msg = tk.StringVar()
        self.EAR_msg.set('1.0')
        self.EAR_box = tk.Message(self.displayFrame, textvariable=self.EAR_msg, relief=tk.GROOVE, justify=tk.CENTER, padx=5)
        self.EAR_box.grid(row=4, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))        

        # Head Position
        self.head_label = tk.Label(self.displayFrame, text='Head Position')
        self.head_label.grid(row=5, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10))        

        self.head_msg = tk.StringVar()
        self.head_msg.set('V Ratio: \nH Ratio: ')
        self.head_box = tk.Message(self.displayFrame, textvariable=self.head_msg, relief=tk.GROOVE, justify=tk.CENTER, padx=5)        
        self.head_box.grid(row=6, column=1, columnspan=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 10)) 

        # Start Button
        self.start_btn = tk.Button(self.displayFrame, text='START', command=self.onStart)
        self.start_btn.grid(row=7, column=1, sticky=tk.N+tk.S+tk.E+tk.W, padx=(10, 0))

        # Stop Button
        self.stop_btn = tk.Button(self.displayFrame, text='STOP', command=self.onStop)
        self.stop_btn.grid(row=7, column=2, sticky=tk.N+tk.S+tk.E+tk.W, padx=(0, 10))


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
            self.status_label.config(fg='Red')
        elif type == 'distraction':
            self.status_msg.set('Distraction ALERT!')
            self.status_label.config(fg='Red')
        else:
            self.status_msg.set('Normal')
            self.status_label.config(fg='Green')


    def quit(self):
        '''
        Clean up the camera and TK windows
        '''
        self._vs.release()
        self._root.quit()



if __name__ == '__main__':

    root = tk.Tk()
    # root.rowconfigure(0,weight=1)
    # root.columnconfigure(0,weight=1)
    root.wm_title("Drowsiness Detector")

    vs = cv2.VideoCapture(0)

    myapp = GUI(root, vs)

    tk.mainloop()



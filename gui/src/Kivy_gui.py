import Tkinter as tk
#from Tkinter import ttk
import subprocess
import time
import rospy
from PIL import Image, ImageTk

from std_msgs.msg import Bool
import cv2
'''
issues: 
- speed up the login process by looking for the $ which shows the login has completed
- add steps for cmd_vel
- add display for heading
'''

class Steps_to_Process(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.name = tk.StringVar()
        self.title_string = tk.StringVar()
        self.title_string.set("Lawn Tractor Startup")
        title_label = ttk.Label(self, textvariable=self.title_string, font=("TkDefaultFont", 12), wraplength=200)

        step_1_button = ttk.Button(self, text="Start", width=50, command=self.step_1_actions)
        self.columnconfigure(0, weight=1)
        step_1_button.grid(row=1, column=0, sticky=tk.W)


# class MyVideoCapture:
#     def __init__(self,video_source=0):
#         self.vid=cv2.VideoCapture(0)
#         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         return self.vid

class ROS_GUI():
    """ROS GUI Main Application"""
    def __init__(self, window):
        self.window = window
        # self.cap = cv2.V
        self.cap = cv2.VideoCapture()
        # self.cap1 = cv2.VideoCapture()
        # self.cap.open("rtsp://admin:Teaminferno@192.168.0.251:554/cam/realmonitor?channel=1&subtype=0")
        self.cap.open("rtsp://admin:Teaminferno@192.168.0.250:554/cam/realmonitor?channel=1&subtype=0")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0);
        # self.cap1.set(cv2.CAP_PROP_BUFFERSIZE, 0);
        # self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=2000, height=1000)
        self.interval = 1 # Interval in ms to get the latest frame

        step_1_button = tk.Button(self.window, text="Start", width=50, command=self.step_1_actions)
        step_1_button.place(x=1200,y=10)
        self.canvas.place(x=0,y=0)
        self.update_image()


    def update_image(self):
        # Get the latest frame and convert image format
        # self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)  # to RGB
        self.image = self.cap.read()[1]
        frame = cv2.resize(self.image, (500,500), interpolation=cv2.INTER_AREA)
        self.image = Image.fromarray(frame)  # to PIL format
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0,anchor=tk.NW, image=self.image)
        # self.update_image()
        # self.image1 = self.cap.read()[1]
        # frame1 = cv2.resize(self.image1, (500,500), interpolation=cv2.INTER_AREA)
        # self.image1 = Image.fromarray(frame1)  # to PIL format
        # self.image1 = ImageTk.PhotoImage(self.image1)
        # self.canvas.create_image(510, 0, anchor=tk.NW, image=self.image1)
        # self.image2 = self.cap.read()[1]
        # frame2 = cv2.resize(self.image2, (500,500), interpolation=cv2.INTER_AREA)
        # self.image2 = Image.fromarray(frame2)  # to PIL format
        # self.image2 = ImageTk.PhotoImage(self.image2)
        # self.canvas.create_image(0, 510, anchor=tk.NW, image=self.image2)
        # self.image3 = self.cap.read()[1]
        # frame3 = cv2.resize(self.image3, (500,500), interpolation=cv2.INTER_AREA)
        # self.image3 = Image.fromarray(frame3)  # to PIL format
        # self.image3 = ImageTk.PhotoImage(self.image3)
        # self.canvas.create_image(510, 510, anchor=tk.NW, image=self.image3)
        # self.window.after(self.interval, self.update_image)
        # self.image5 = self.cap.read()[1]
        # frame = cv2.resize(self.image5, (800, 800), interpolation=cv2.INTER_AREA)
        # self.image5 = Image.fromarray(frame)  # to PIL format
        # self.image5 = ImageTk.PhotoImage(self.image5)
        # self.canvas.create_image(1020, 100, anchor=tk.NW, image=self.image5)

    def step_1_actions(self):
        pub = rospy.Publisher('/button', Bool, queue_size=1)
        rospy.init_node('simple_gui', anonymous=True)

        print("button pressed")
        msg = True
        pub.publish(msg)

        

if __name__ == '__main__':
    root = tk.Tk()
    ROS_GUI(root)
    root.mainloop()


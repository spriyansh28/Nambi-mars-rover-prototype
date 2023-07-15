import tkinter as tk
from tkinter import ttk
import subprocess
import time
import rospy
from std_msgs.msg import Bool
# import cv2
'''
issues: 
- speed up the login process by looking for the $ which shows the login has completed
- add steps for cmd_vel
- add display for heading
'''

class Steps_to_Process(tk.Frame):
    def _init_(self, parent, *args, **kwargs):
        super()._init_(parent, *args, **kwargs)

        self.name = tk.StringVar()
        self.title_string = tk.StringVar()
        self.title_string.set("Lawn Tractor Startup")
        title_label = ttk.Label(self, textvariable=self.title_string, font=("TkDefaultFont", 12), wraplength=200)

        step_1_button = ttk.Button(self, text="Start", width=50, command=self.step_1_actions)
        self.columnconfigure(0, weight=1)
        step_1_button.grid(row=1, column=0, sticky=tk.W)
    def step_1_actions(self):
        pub = rospy.Publisher('/button', Bool, queue_size=1)
        rospy.init_node('simple_gui', anonymous=True)

        print("button pressed")
        msg = True
        pub.publish(msg)

# class MyVideoCapture:
#     def _init_(self,video_source=0):
#         self.vid=cv2.VideoCapture(0)
#         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         return self.vid

class ROS_GUI(tk.Tk):
    """ROS GUI Main Application"""
    def _init_(self, *args, **kwargs):
        super()._init_(*args, **kwargs)

        # set the window properties
        self.title("ROS GUI")
        self.geometry("200x300")
        self.resizable(width=False, height=False)
        # self.vid = MyVideoCapture(self.video_source)
        # self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        # # Define the UI
        Steps_to_Process(self).grid(sticky=(tk.E + tk.W + tk.N + tk.S))
        self.columnconfigure(0, weight=1)
if __name__ == '__main__':
    app = ROS_GUI()
    app.mainloop()
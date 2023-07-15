#!/usr/bin/env python
import rospy
from pcl_msgs.msg import Vertices
import time
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


pantilt = Vertices()
pan = 90
tilt = 90
pantilt.vertices=[90, 90]

def find_pwntest4(msg):
    global pan
    global tilt
    if( msg.axes[6] > 0) :
        pan=pan+0.5
    elif(msg.axes[6] < 0) :
        if pan>0:
            pan=pan-0.5
    elif(msg.axes[7] > 0) :
        tilt=tilt+0.5
    elif(msg.axes[7] < 0) :
        if tilt>0:
            tilt=tilt-0.5
    return pan,tilt        

def find_pwmtest5(msg):
    left_speed = 0
    right_speed = 0
    
    if msg.buttons[3] == 1:
        left_speed= 0
        right_speed = 0           
    elif msg.axes[1] > 0 :
        left_speed= msg.axes[1] * (255)
        right_speed = msg.axes[1] * (255)
        if msg.axes[3] < 0 :
            left_speed= msg.axes[1] * (255)
            right_speed =  msg.axes[1] * (50)
        elif msg.axes[3] > 0 :
            left_speed=  msg.axes[1] * (50)
            right_speed = msg.axes[1] * (255)

    elif msg.axes[1] < 0 :
        left_speed= msg.axes[1] * (255)
        right_speed = msg.axes[1] * (255)
        if msg.axes[3] < 0 :
            left_speed= msg.axes[1] * (-255)
            right_speed = msg.axes[1] * (-50)
        elif msg.axes[0] > 0 :
            left_speed= msg.axes[1] * (-50)
            right_speed = msg.axes[1] * (-255)

    elif msg.axes[3] > 0 :
        left_speed= msg.axes[3] * (-255)
        right_speed = msg.axes[3] * (255)

    elif msg.axes[3] < 0 :
        left_speed = msg.axes[3] * (-255)
        right_speed = msg.axes[3] * (255)

    elif msg.buttons[5] == 1:
        left_speed= -5
        right_speed = -5   

    elif msg.buttons[4] == 1:
        left_speed= 5
        right_speed = 5  
    
    
    linear_vel  = (left_speed + right_speed) / 2 # (m/s)
    angular_vel  = (right_speed - left_speed) / 2 # (rad/s)
    return linear_vel,angular_vel

def find_twist(data):
    drive_com = Twist()
    pan, tilt = find_pwntest4(data)
    pantilt.vertices[1] = tilt
    pantilt.vertices[0] = pan
    linear_vel, angular_vel=find_pwmtest5(data)
    drive_com.linear.x = float(linear_vel)
    drive_com.angular.z = float(angular_vel)
    return drive_com

    
def callback(msg):
    rate = rospy.Rate(200)
    drive_com = find_twist(msg)
    pub = rospy.Publisher('servo_pantilt', Vertices , queue_size=10)
    pub.publish(pantilt)
    print("published2")
    pub = rospy.Publisher('cmd_vel', Twist,queue_size=10) 
    pub.publish(drive_com)
    print("published1")
   
    rate.sleep()
    #rospy.loginfo("I heard %f",msg.buttonss[])
    

def listener():
    rospy.init_node('listener_joy', anonymous=True)
    rospy.Subscriber("j0", Joy, callback)
    rospy.spin()

if __name__ == '__main__':
    
    pan=90
    tilt=90
    left_speed = 0
    right_speed =0

    listener()


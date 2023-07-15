#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
from numpy import interp
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

def find_pwmtest5(msg):
    left_speed = 0
    right_speed = 0
    if msg.buttons[1]>0:
        left_speed=0
        right_speed=0
    else:
        if msg.axes[1] > 0 :
            left_speed= msg.axes[1] * (-255)
            right_speed = msg.axes[1] * (-255)
            if msg.axes[3] > 0 :
                left_speed= msg.axes[5] * (-255)
                right_speed = 0
            elif msg.axes[3] < 0 :
                left_speed= 0
                right_speed = msg.axes[5] * (-255)
        elif msg.axes[1] < 0 :
            left_speed= msg.axes[1] * (255)
            right_speed = msg.axes[1] * (255)
            if msg.axes[3] > 0 :
                left_speed= msg.axes[2] * (255)
                right_speed = 0
            elif msg.axes[3] < 0 :
                left_speed= 0
                right_speed = msg.axes[2] * (255)
        elif msg.axes[2] < 1 and msg.axes[2] > -1 :
            left_speed= msg.axes[0] * (-255)
            right_speed = -0.2
        elif msg.axes[5] < 1 and msg.axes[5]>-1 :
            left_speed = -0.2
            right_speed = msg.axes[0] * (255)

    
    linear_vel  = (left_speed + right_speed) / 2 # (m/s)
    angular_vel  = (right_speed - left_speed) / 2 # (rad/s)
    return linear_vel,angular_vel

def find_twist(data):
    drive_com = Twist()
    linear_vel, angular_vel = find_pwmtest5(data)
    drive_com.linear.x = float(linear_vel)
    drive_com.angular.z = float(angular_vel)
    return drive_com

    
def callback(msg):
    rate = rospy.Rate(200)
    drive_com = find_twist(msg)
    pub = rospy.Publisher('cmd_vel', Twist,queue_size=1)  
    pub.publish(drive_com)
    print("published")
    rate.sleep()
    #rospy.loginfo("I heard %f",msg.buttons[])
    

def listener():
    rospy.init_node('listener_joy', anonymous=True)
    rospy.Subscriber("joy", Joy, callback)
    rospy.spin()

if __name__ == '__main__':

    left_speed = 0
    right_speed =0

    listener()


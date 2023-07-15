#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
from numpy import interp
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


 #if turned on the direction is reversed






def find_pwmtest5(msg):
    
     left_speed = msg.axes[1] *255
     right_speed = msg.axes[4]*255


    stop = msg.buttons[1]
    if(stop==1):
        linear_speed=0
        angular_speed=0
  linear_vel  = (left_speed + right_speed) / 2 # (m/s)
  angular_vel  = (right_speed - left_speed) / 2 # (rad/s)
    return linear_speed,angular_speed

def find_twist(data):
    drive_com = Twist()
    linear_vel, angular_vel = find_pwmtest5(data)
    drive_com.linear.x = float(linear_vel)
    drive_com.angular.z = float(angular_vel)
    return drive_com

    

#def transform(a):
#    point.x = pwm[0] 
#    point.y = pwm[1]
#    point.z = pwm[2]  

def callback(msg):
    rate = rospy.Rate(200)
    drive_com = find_twist(msg)
    pub = rospy.Publisher('cmd_vel', Twist,queue_size=1)  
    pub.publish(drive_com)
    rate.sleep()
    #rospy.loginfo("I heard %f",msg.buttons[])
    

def listener():
    rospy.init_node('listener_joy', anonymous=True)
    rospy.Subscriber("joy", Joy, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

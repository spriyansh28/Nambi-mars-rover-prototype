#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
from numpy import interp
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


act1 = 90
act2 = 90




def find_angles(msg):
    global act1, act2
    if(msg.buttons[3] == 1):
        act1 += 5
    elif(msg.buttons[1] == 1):
        act1 -= 5
    elif(msg.buttons[2] == 1):
        act2 += 5
    elif(msg.buttons[0] == 1):
        act2 -= 5


    return act2, act1



def callback(msg):
    rate = rospy.Rate(200)
    arm_angles = JointState()
    act1, act2 = find_angles(msg)
    arm_angles.position.append(act1)
    arm_angles.position.append(act2)
    pub = rospy.Publisher('joint_states', JointState,queue_size=1)   
    pub.publish(arm_angles)
    rate.sleep()
    # rospy.loginfo("I heard %f",msg.buttons[])
    

def listener():
    rospy.init_node('listener_joy', anonymous=True)
    rospy.Subscriber("joy", Joy, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

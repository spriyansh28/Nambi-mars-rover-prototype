#include<ros.h>
#include <std_msgs/String.h>
#include<geometry_msgs/Twist.h>
  

 
ros::NodeHandle nh;
const int red =  3;// put red wire on 5
const int green=5;//put green wire on 3

 


 void blnk(const geometry_msgs::Twist msg)
 {
    if(msg.linear.x)
    {
      digitalWrite(green,LOW);
      delay(10);
      digitalWrite(red,HIGH);  
    }
    else 
    {
      digitalWrite(red,LOW);
      delay(10);
      digitalWrite(green,HIGH);
    }
 }

 ros::Subscriber <geometry_msgs::Twist> sub("/JOY_node",100,&blnk);
 
 void setup() 
 {
  // set the digital pin as output:
  pinMode(green, OUTPUT);
  pinMode(red, OUTPUT);
  nh.initNode();
  nh.subscribe(sub);
 }

void loop() 
{

  nh.spinOnce();
  delay(1);
}

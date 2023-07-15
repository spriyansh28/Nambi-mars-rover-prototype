/*
 * rosserial Servo Control Example
 *
 * This sketch dmonstrates the control of hobby R/C servos
 * using ROS and the arduiono
 * 
 * For the full tutorial write up, visit
 * www.ros.org/wiki/rosserial_arduino_demos
 *
 * For more information on the Arduino Servo Library
 * Checkout :
 * http://www.arduino.cc/en/Reference/Servo
 */

//#if (ARDUINO >= 100)
// #include <Arduino.h>
//#else
// #include <WProgram.h>
//#endif


#include <Servo.h> 
#include <ros.h>
#include <arm/Pwm.h>
#include <std_msgs/UInt16.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/JointState.h>

ros::NodeHandle  nh;

 int pwm1pin = 13, pwm2pin = 12, dir1,dir2;
  int dirpin1=45, dirpin2=34;
  int xpotpin = A0, ypotpin = A1; 

void set_length( const arm::Pwm& cmd_msg)
{
  int x = cmd_msg.shoulder_angle;
  int y = cmd_msg.elbow_angle;
  //int z = cmd_msg.z;

 

//   int xmap = map(x, 0, 90, 285, 940);
//  int ymap = map(y, 0, 60, 60, 980); 
    int xmap = map(x, 0, 90, 80, 1013);
    int ymap = map(y, 0, 60, 992, 25); 
  //int zmap = map(z, 0, 360, 993, 55);

  int xval = analogRead(xpotpin);
  //Serial.println(xval);
  int yval = analogRead(ypotpin);
  //Serial.println(yval);
  //int zval = analogRead(zpotpin);

  if(xval<xmap)
  {
    dir1 = 0;
    //Serial.println("Increasing");
  }
  else if(xval>xmap)
  {
    dir1 = 1;
    //Serial.println("Decreasing");
  }

  if(yval<ymap)
  {
    dir2 = 0;
  }
  else if(yval>ymap)
  {
    dir2 = 1;
  }

  /*if(zval>zmap)
  {
    dir3 = 1;
  }
  else
  {
    dir3=0;
  }*/
  
  while((!(xval<(xmap+30) && xval>(xmap-30))) || (!(yval<(ymap+30) && yval>(ymap-30)) ))
  {
    xval = analogRead(xpotpin);
    yval = analogRead(ypotpin);
    
    //analogWrite(pwm3pin, 255);
   
    //digitalWrite(dirpin3, dir3);
    
    if(xval<(xmap+30) && xval>(xmap-30))
    {
      analogWrite(pwm1pin, 0);
    }
    else
    {
      analogWrite(pwm1pin, 255);
      digitalWrite(dirpin1, dir1);
      //Serial.println("Here");
      //Serial.println(yval);
      
    }

   if(yval<(ymap+30) && yval>(ymap-30))
    {
      analogWrite(pwm2pin, 0);
    }
    else
    {
      analogWrite(pwm2pin, 255);
      digitalWrite(dirpin2, dir2);
      //Serial.println("Here2");
      //Serial.println(xval);
      
    }
    /*if(zval == zmap)
    {
      analogWrite(pwm3pin, 0);
    }
    else
    {
      analogWrite(pwm1pin, 255);
      digitalWrite(dirpin1, dir1);
      
    }*/
    
  }
  
  /*int x = map(l, 20, 120, 993, 55);
  int dir;
  int val = analogRead(A0);
  Serial.print("Sensor reading :");
  Serial.print(val);
  Serial.println("\n");
  if(val>x)
  {
    dir =1;
  }
  else
  {
    dir = 0;
  }
  while(val!=x)
  {
    val=analogRead(A0);
    analogWrite(pwmpin, 255);
    digitalWrite(dirpin, dir); 
  }
    analogWrite(pwmpin, 0);
    digitalWrite(dirpin , 0);*/
}



void servo_cb( const arm::Pwm& cmd_msg){
  digitalWrite(13, HIGH-digitalRead(13));
   set_length(cmd_msg);  
    //toggle led  
}

ros::Subscriber<arm::Pwm> sub("set", servo_cb );

void setup(){
  pinMode(13, OUTPUT);
  pinMode(pwm1pin, OUTPUT);
  pinMode(dirpin1, OUTPUT);
   pinMode(pwm2pin, OUTPUT);
  pinMode(dirpin2, OUTPUT);
  pinMode(xpotpin, INPUT);
  pinMode(ypotpin, INPUT);
  analogWrite(pwm1pin, 0);
  analogWrite(pwm2pin, 0);
  nh.initNode();
  nh.subscribe(sub);
 // Serial.begin(9600);

 
 
   //attach it to pin 9
}

void loop(){
  nh.spinOnce();
  delay(1);
}

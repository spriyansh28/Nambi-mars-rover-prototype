/*
  Gearmotor Rotary Encoder Test
  motor-encoder-rpm.ino
  Read pulses from motor encoder to calculate speed
  Control speed with potentiometer
  Displays results on Serial Monitor
  Use Cytron MD10C PWM motor controller
  DroneBot Workshop 2019
  https://dronebotworkshop.com
*/
 #include <ros.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Int16MultiArray.h>

// Motor encoder output pulse per rotation (change as required)
#define ENC_COUNT_REV 696
 
// Encoder output to Arduino Interrupt pin
#define ENCA 2  //A pin 
 #define ENCB 3
// MD10C PWM connected to pin 10
#define PWMA 4
#define PWMB 5 
// MD10C DIR connected to pin 12
#define DIRA 11
#define DIRB 12 

float track=0.45;
float VL;
float VR;

ros::NodeHandle nh;

geometry_msgs::Twist str_message;
std_msgs::Int16MultiArray str_msg;
ros::Publisher chatter("feedback", &str_msg);
//ros::Publisher chatter("velocity", &str_message);
void Motor1L(float motion);
void Motor1R(float motion);

float speed_angular=0, speed_linear=0;
void messageCb( const geometry_msgs::Twist& msg){
  speed_angular = msg.angular.z;
  speed_linear = msg.linear.x;
  speed_angular = (speed_angular*track)/(2.0);
  VR=speed_linear + speed_angular;
  VL= speed_linear - speed_angular;

   Motor1L(VL);
   Motor1R(VR);

}

ros::Subscriber<geometry_msgs::Twist> sub("cmd_vel", &messageCb );
 
// Analog pin for potentiometer
int speedcontrol = 0;
int data[2] ={0,0};
// Pulse count from encoder
volatile long encoderValueA = 0;
 volatile long encoderValueB = 0;
// One-second interval for measurements
int interval = 1000;
 
// Counters for milliseconds during interval
long previousMillis = 0;
long currentMillis = 0;
 
// Variable for RPM measuerment
int rpmA = 0;
int rpmB = 0; 
// Variable for PWM motor speed output
int motorPwm = 0;
 
void setup()
{
  // Setup Serial Monitor
//  Serial.begin(57600); /
  
  // Set encoder as input with internal pullup  
  pinMode(ENCA, INPUT_PULLUP); 
 pinMode(ENCB, INPUT_PULLUP); 
  // Set PWM and DIR connections as outputs
  pinMode(PWMA, OUTPUT);
  pinMode(DIRA, OUTPUT);
  pinMode(PWMB, OUTPUT);
  pinMode(DIRB, OUTPUT);
  
  // Attach interrupt 
  attachInterrupt(digitalPinToInterrupt(ENCA), updateEncoderA, RISING);
  attachInterrupt(digitalPinToInterrupt(ENCB), updateEncoderB, RISING);
  digitalWrite(PWMA, LOW);
  digitalWrite(PWMB, LOW);
  // Setup initial values for timer
  previousMillis = millis();

  nh.initNode();
  nh.advertise(chatter);
 nh.subscribe(sub);
}
 
void loop()
{
  
    // Control motor with potentiometer
    //motorPwm = 200;
    
    // Write PWM to controller
    //analogWrite(PWM, motorPwm);
  
  // Update RPM value every second
  currentMillis = millis();
  if (currentMillis - previousMillis > interval) {
    previousMillis = currentMillis;
 
 
    // Calculate RPM
    rpmA = (float)(encoderValueA * 60 / ENC_COUNT_REV);
    rpmB = (float)(encoderValueB * 60 / ENC_COUNT_REV);
    str_message.linear.x = rpmA*((3.14*0.15)/30);
    str_message.linear.y = rpmB*((3.14*0.15)/30);
 
    // Only update display when there is a reading
      data[0]=encoderValueB;
      data[1]=rpmB;
      str_msg.data[0] = encoderValueA;
      str_msg.data[1] = rpmA;
      str_msg.data[2] = encoderValueB;
      str_msg.data[3] = rpmB;

      str_msg.data_length = 4;
      Serial.print("PWM VALUE: ");
      Serial.print(motorPwm);
      Serial.print('\t');
      Serial.print(" EN A: ");
      Serial.print(encoderValueA);
      Serial.print('\t');

      Serial.print("EN B: ");
      Serial.print(encoderValueB);

      Serial.print('\t');
      Serial.print(" RPM A: ");
      Serial.print(rpmA);
            Serial.print('\t');

      Serial.print(" RPM B: ");

      Serial.print(rpmB);
            Serial.print('\t');


      Serial.println(" RPM");

   
    encoderValueA = 0;
    encoderValueB = 0;
  }
chatter.publish( &str_msg );

  nh.spinOnce();
 delay(100);
}
 
void updateEncoderA()
{

  // Increment value for each pulse from encoder
  if(digitalRead(4)==0){
    encoderValueA++;  
  }else{
    encoderValueA--;
  }
}
void updateEncoderB()
{
  // Increment value for each pulse from encoder
  if(digitalRead(5)==0){
    encoderValueB++;  
  }else{
    encoderValueB--;
  }
}

void Motor1L(float motion){
if(motion>=0){
  analogWrite(PWMA, motion);
  digitalWrite(DIRA, HIGH); //clockwise
}else{
  motion=-1*motion;
  analogWrite(PWMA, motion);
  digitalWrite(DIRA, LOW); //anti-clockwise
}

}
void Motor1R(float motion){
if(motion>=0){
  analogWrite(PWMB, motion);
  digitalWrite(DIRB, HIGH); //clockwise
}else{
  motion=-1*motion;
  analogWrite(PWMB, motion);
  digitalWrite(DIRB, LOW); //anti-clockwise
}

}
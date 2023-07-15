#include <ros.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Int16.h>



// Motor encoder output pulse per rotation (change as required)
#define ENC_COUNT_REV 22600
 
// Encoder output to Arduino Interrupt pin
#define ENCA 2  //A pin 
#define ENCA2 4  //A pin 
#define ENCB 3
#define ENCB2 7

// MD10C PWM connected to pin 10
#define PWMA 6
#define PWMB 5 
// MD10C DIR connected to pin 12
#define DIRA 9
#define DIRB 11

float track=0.45;
float VL;
float VR;

ros::NodeHandle nh;

geometry_msgs::Twist str_message;
ros::Publisher chatter("velocity", &str_message);

std_msgs::Int16 right_wheel_tick_count;
ros::Publisher rightPub("right_ticks", &right_wheel_tick_count);
 
std_msgs::Int16 left_wheel_tick_count;
ros::Publisher leftPub("left_ticks", &left_wheel_tick_count);



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
  Serial.begin(9600); 
  
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
  nh.subscribe(sub);
  nh.advertise(rightPub);
  nh.advertise(leftPub);
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

    right_wheel_tick_count.data = encoderValueA;
    left_wheel_tick_count.data =encoderValueA;
    // Only update display when there is a reading
    

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

      Serial.println(rpmB);
            Serial.print('\t');


    rightPub.publish( &right_wheel_tick_count );
    leftPub.publish( &left_wheel_tick_count );
     nh.spinOnce();

    encoderValueA = 0;
    encoderValueB = 0;
  }

 delay(1);
}
 
void updateEncoderA()
{

  // Increment value for each pulse from encoder
  if(digitalRead(ENCA2)==0){
    encoderValueA++;  
  }else{
    encoderValueA--;
  }
}
void updateEncoderB()
{
  // Increment value for each pulse from encoder
  if(digitalRead(ENCB2)==0){
    encoderValueB++;  
  }else{
    encoderValueB--;
  }
}

void Motor1L(float motion){
if(motion>=0){
 // analogWrite(PWMA, motion);
  digitalWrite(DIRA, HIGH); //clockwise
}else{
  motion=-1*motion;
  //analogWrite(PWMA, motion);
  digitalWrite(DIRA, LOW); //anti-clockwise
}

}
void Motor1R(float motion){
if(motion>=0){
 // analogWrite(PWMB, motion);
  digitalWrite(DIRB, HIGH); //clockwise
}else{
  motion=-1*motion;
 // analogWrite(PWMB, motion);
  digitalWrite(DIRB, LOW); //anti-clockwise
}

}

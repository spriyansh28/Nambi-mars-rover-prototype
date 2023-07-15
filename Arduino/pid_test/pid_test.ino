#include <util/atomic.h>
 
#define ENCA1 2
#define ENCA2 4 
#define ENCB1 3
#define ENCB2 7 
#define PWMA 6
#define PWMB 5
#define DIRA 9
#define DIRB 10
 
volatile int posi = 0; 
long prevT = 0;
float eprev = 0;
float eintegral = 0;

volatile int posi2 = 0; 
long prevT2 = 0;
float eprev2 = 0;
float eintegral2 = 0;
 
void setup() {
  Serial.begin(9600);
  pinMode(ENCA1,INPUT);
  pinMode(ENCB1,INPUT);
  pinMode(ENCA2,INPUT);
  pinMode(ENCB2,INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCA1),readEncoder,RISING);
  attachInterrupt(digitalPinToInterrupt(ENCB1),readEncoder2,RISING);
  
  pinMode(PWMA,OUTPUT);
  pinMode(PWMB,OUTPUT);
  pinMode(DIRA,OUTPUT);
 // pinMode(INA2,OUTPUT);
  pinMode(DIRB,OUTPUT);
//  pinMode(INB2,OUTPUT);
  
  Serial.println("target pos");
}
 
void loop() {
 
 
  int target = 250*sin(prevT/1e6);
  int target2 = 250*sin(prevT2/1e6);
 
  // PID constants
  float kp = 1;
  float kd = 0.025;
  float ki = 0.0;
 
  // time difference
  long currT = micros();
  long currT2 = micros();
  float deltaT = ((float) (currT - prevT))/( 1.0e6 );
  float deltaT2 = ((float) (currT2 - prevT2))/( 1.0e6 );
  prevT = currT;
  prevT2 = currT2;
 
  int pos = 0; 
  int pos2 = 0; 
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    pos = posi;
    pos2 = posi2;
  }
  
  // error
  int e = pos - target;
  int e2 = pos2 - target2;
 
  // derivative
  float dedt = (e-eprev)/(deltaT);
  float dedt2 = (e2-eprev2)/(deltaT2);
 
  // integral
  eintegral = eintegral + e*deltaT;
   eintegral2 = eintegral2 + e2*deltaT2;
 
  // control signal
  float u = kp*e + kd*dedt + ki*eintegral;
  float u2 = kp*e2 + kd*dedt2 + ki*eintegral2;
 
  // motor power
  float pwr = fabs(u);
  float pwr2 = fabs(u2);
  if( pwr > 255 ){
    pwr = 255;
  }
  if( pwr2 > 255 ){
    pwr2 = 255;
  }
 
  // motor direction
  int dir = 1;
  int dir2 = 1;
  if(u<0){
    dir = -1;
  }
  if(u2<0){
    dir2 = -1;
  }
 
  // signal the motor
  setMotor(dir,pwr,PWMA,DIRA);
  setMotor(dir2,pwr2,PWMB,DIRB);
 
 
  // store previous error
  eprev = e;
  eprev2 = e2;
 
  Serial.print(target);
  Serial.print(" ");
  Serial.print(pos);
  Serial.println();

  Serial.print(target2);
  Serial.print(" ");
  Serial.print(pos2);
  Serial.println();
}
 
void setMotor(int dir, int pwmVal, int pwm, int in){
  analogWrite(pwm,pwmVal);
  if(dir == 1){
    digitalWrite(in,HIGH);
  }
  else if(dir == -1){
    digitalWrite(in,LOW);
  }
  else{
    digitalWrite(in,LOW);
  }  
}
 
void readEncoder(){
  int b = digitalRead(ENCA2);
  if(b > 0){
    posi++;
  }
  else{
    posi--;
  }
}
void readEncoder2(){
  int b = digitalRead(ENCB2);
  if(b > 0){
    posi2++;
  }
  else{
    posi2--;
  }
}

#include<ros.h>
#include<std_msgs/Float32.h>
#include <std_msgs/Int16.h>
#include <geometry_msgs/Point.h>
#include "Adafruit_CCS811.h"
#include <Wire.h>
#include "Adafruit_BME680.h"

#define BME_SCK 13
#define BME_MISO 12
#define BME_MOSI 11
#define BME_CS 10
#include <std_msgs/Int16.h>
#include <geometry_msgs/Point.h>

#define SEALEVELPRESSURE_HPA (1013.25)
#include <Adafruit_Sensor.h>
#include <MQUnifiedsensor.h>
#include <Adafruit_TSL2561_U.h>
#include "SparkFun_AS7265X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS7265X
AS7265X sensor;

#define BME_SCK 13  //bme
#define BME_MISO 12  //bme
#define BME_MOSI 11  //bme
#define BME_CS 10   //bme

#define pin_mq2 A1   //mq2
#define RO_CLEAN_AIR_FACTOR (9.21)//RO_CLEAR_AIR_FACTOR=(Sensor resistance in clean ai    //mq2
float RO=10;// initial resistance value    //mq2



#define placa "Arduino UNO"  //mq131
#define Voltage_Resolution 5  //mq131
#define pin A2 //Analog input 0 of your arduino  //mq131
#define type "MQ-131" //MQ131
#define ADC_Bit_Resolution 10 // For arduino UNO/MEGA/NANO  //mq131
#define RatioMQ131CleanAir 15 //RS / R0 = 15 ppm   //mq131

const float VRefer = 3.3;       // O2
 const int pinAdc   = A0;     //O2

 unsigned long int avgValue;  //ph
float b;                    //ph
int buf[10],temp;        //ph
 #define SensorPin A4   //ph

ros::NodeHandle  nh;
std_msgs::Float32 lumin_msg;
ros::Publisher chatter1("luminometer", &lumin_msg);
std_msgs::Float32 ccs_msg;
ros::Publisher chatter2("ccs811", &ccs_msg);
Adafruit_CCS811 ccs;
std_msgs::Float32 bme_msg;
ros::Publisher chatter3("bme680", &bme_msg);
Adafruit_BME680 bme;
std_msgs::Float32 spectro_msg;
ros::Publisher chatter4("spectrometer", &spectro_msg);
geometry_msgs::Point mq2_msg; 
ros::Publisher chatter5("Gas_SENSOR", &mq2_msg);
std_msgs::Float32 mq131_msg;
ros::Publisher chatter6("mq131",&mq131_msg);
MQUnifiedsensor MQ131(placa, Voltage_Resolution, ADC_Bit_Resolution, pin, type);
std_msgs::Float32 O2_msg;
ros::Publisher chatter7("oxygen", &O2_msg);
std_msgs::Float32 ph_msg;
ros::Publisher chatter8("phsensor", &ph_msg);
std_msgs::Float32 uv_msg;
ros::Publisher chatter9("uvsensor", &uv_msg);
 



/* This driver uses the Adafruit unified sensor library (Adafruit_Sensor),
   which provides a common 'type' for sensor data and some helper functions.
   
   To use this driver you will also need to download the Adafruit_Sensor
   library and include it in your libraries folder.

   You should also assign a unique ID to this sensor for use with
   the Adafruit Sensor API so that you can identify this particular
   sensor in any data logs, etc.  To assign a unique ID, simply
   provide an appropriate value in the constructor below (12345
   is used by default in this example).
   
   Connections
   ===========
   Connect SCL to I2C SCL Clock
   Connect SDA to I2C SDA Data
   Connect VCC/VDD to 3.3V or 5V (depends on sensor's logic level, check the datasheet)
   Connect GROUND to common ground

   I2C Address
   ===========
   The address will be different depending on whether you leave
   the ADDR pin floating (addr 0x39), or tie it to ground or vcc. 
   The default addess is 0x39, which assumes the ADDR pin is floating
   (not connected to anything).  If you set the ADDR pin high
   or low, use TSL2561_ADDR_HIGH (0x49) or TSL2561_ADDR_LOW
   (0x29) respectively.
    
   History
   =======
   2013/JAN/31  - First version (KTOWN)
*/
   
Adafruit_TSL2561_Unified tsl = Adafruit_TSL2561_Unified(TSL2561_ADDR_FLOAT, 12345);

/**************************/
/*
    Displays some basic information on this sensor from the unified
    sensor API sensor_t type (see Adafruit_Sensor for more information)
*/
/**************************/
void displaySensorDetails(void)
{
  
  //luminometer
  sensor_t sensor;
  tsl.getSensor(&sensor);
  Serial.println("------------------------------------");
  Serial.print  ("Sensor:       "); Serial.println(sensor.name);
  Serial.print  ("Driver Ver:   "); Serial.println(sensor.version);
  Serial.print  ("Unique ID:    "); Serial.println(sensor.sensor_id);
  Serial.print  ("Max Value:    "); Serial.print(sensor.max_value); Serial.println(" lux");
  Serial.print  ("Min Value:    "); Serial.print(sensor.min_value); Serial.println(" lux");
  Serial.print  ("Resolution:   "); Serial.print(sensor.resolution); Serial.println(" lux");  
  Serial.println("------------------------------------");
  Serial.println("");
  delay(500);
}

/**************************/
/*
    Configures the gain and integration time for the TSL2561
*/
/**************************/
void configureSensor(void)
{
  /* You can also manually set the gain or enable auto-gain support */
  // tsl.setGain(TSL2561_GAIN_1X);      /* No gain ... use in bright light to avoid sensor saturation */
  // tsl.setGain(TSL2561_GAIN_16X);     /* 16x gain ... use in low light to boost sensitivity */
  tsl.enableAutoRange(true);            /* Auto-gain ... switches automatically between 1x and 16x */
  
  /* Changing the integration time gives you better sensor resolution (402ms = 16-bit data) */
  tsl.setIntegrationTime(TSL2561_INTEGRATIONTIME_13MS);      /* fast but low resolution */
  // tsl.setIntegrationTime(TSL2561_INTEGRATIONTIME_101MS);  /* medium resolution and speed   */
  // tsl.setIntegrationTime(TSL2561_INTEGRATIONTIME_402MS);  /* 16-bit data but slowest conversions */

  /* Update these values depending on what you've set above! */  
  Serial.println("------------------------------------");
  Serial.print  ("Gain:         "); Serial.println("Auto");
  Serial.print  ("Timing:       "); Serial.println("13 ms");
  Serial.println("------------------------------------");
}

/**************************/
/*
    Arduino setup function (automatically called at startup)
*/
/**************************/
void setup(void) 
{
  nh.initNode();
  nh.advertise(chatter1);
  nh.advertise(chatter2);
  nh.advertise(chatter3);
  nh.advertise(chatter4);
  nh.advertise(chatter5); // no code in void setup for mq2
  nh.advertise(chatter6);
  nh.advertise(chatter7);
  nh.advertise(chatter8);
  nh.advertise(chatter9);
  Serial.begin(9600);
  Serial.println("Light Sensor Test"); Serial.println("");
  
  /* Initialise the sensor */
  //use tsl.begin() to default to Wire, 
  //tsl.begin(&Wire2) directs api to use Wire2, etc.
  if(!tsl.begin())
  {
    /* There was a problem detecting the TSL2561 ... check your connections */
    Serial.print("Ooops, no TSL2561 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  
  /* Display some basic information on this sensor */
  displaySensorDetails();
  
  /* Setup the sensor gain and integration time */
  configureSensor();
  
  /* We're ready to go! */
  Serial.println("");

  //ccs ccs

  
Serial.println("CCS811 test");
if(!ccs.begin()){
Serial.println("Failed to start sensor! Please check your wiring.");
while(1);
}
//calibrate temperature sensor
while(!ccs.available());
float temp = ccs.calculateTemperature();
ccs.setTempOffset(temp - 25.0);

//bme bme 

while (!Serial);
  Serial.println(F("BME680 test"));

  if (!bme.begin()) {
    Serial.println("Could not find a valid BME680 sensor, check wiring!");
    while (1);
  }

  // Set up oversampling and filter initialization
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150); // 320*C for 150 ms

  //spectro spectro 

  Serial.println("AS7265x Spectral Triad Example");

  if (sensor.begin() == false)
  {
    Serial.println("Sensor does not appear to be connected. Please check wiring. Freezing...");
    while (1)
      ;
  }

  Serial.println("A,B,C,D,E,F,G,H,R,I,S,J,T,U,V,W,K,L");

  //mq131 mq131

  Serial.begin(9600); //Init serial port

  //Set math model to calculate the PPM concentration and the value of constants
  MQ131.setRegressionMethod(1); //_PPM =  a*ratio^b
  MQ131.setA(23.943); MQ131.setB(-1.11); // Configurate the ecuation values to get O3 concentration

  /*
    Exponential regression:
  GAS     | a      | b
  NOx     | -462.43 | -2.204
  CL2     | 47.209 | -1.186
  O3      | 23.943 | -1.11
  */


  /*****************************  MQ Init ********************************************/ 
  //Remarks: Configure the pin of arduino as input.
  /************************************************************************************/ 
  MQ131.init(); 
  /* 
    //If the RL value is different from 10K please assign your RL value with the following method:
    MQ131.setRL(10);
  */
  /*****************************  MQ CAlibration ********************************************/ 
  // Explanation: 
  // In this routine the sensor will measure the resistance of the sensor supposing before was pre-heated
  // and now is on clean air (Calibration conditions), and it will setup R0 value.
  // We recomend execute this routine only on setup or on the laboratory and save on the eeprom of your arduino
  // This routine not need to execute to every restart, you can load your R0 if you know the value
  // Acknowledgements: https://jayconsystems.com/blog/understanding-a-gas-sensor
  Serial.print("Calibrating please wait.");
  float calcR0 = 0;
  for(int i = 1; i<=10; i ++)
  {
    MQ131.update(); // Update data, the arduino will be read the voltage on the analog pin
    calcR0 += MQ131.calibrate(RatioMQ131CleanAir);
    Serial.print(".");
  }
  MQ131.setR0(calcR0/10);
  Serial.println("  done!.");
  
  if(isinf(calcR0)) {Serial.println("Warning: Conection issue founded, R0 is infite (Open circuit detected) please check your wiring and supply"); while(1);}
  if(calcR0 == 0){Serial.println("Warning: Conection issue founded, R0 is zero (Analog pin with short circuit to ground) please check your wiring and supply"); while(1);}
  /*****************************  MQ CAlibration ********************************************/ 
  MQ131.serialDebug(true);

  //o2 o2

  Serial.println("Grove - Oxygen Sensor(MIX8410) Test Code...");

  //ph ph 
  pinMode(13,OUTPUT);  
   Serial.println("Ready");
}

/**************************/
/*
    Arduino loop function, called once 'setup' is complete (your own code
    should go here)
*/
/**************************/


void loop(void) 
{  
  /* Get a new sensor event */ 
  sensors_event_t event;
  tsl.getEvent(&event);
 
  /* Display the results (light is measured in lux) */
  if (event.light)
  {
    //Serial.print(event.light); Serial.println(" lux");
    lumin_msg.data=event.light;
    chatter1.publish(&lumin_msg);
  }
  else
  {
    /* If event.light = 0 lux the sensor is probably saturated
       and no reliable data could be generated! */
    Serial.println("Sensor overload");
  }

    //ccs ccs
    
    if(ccs.available()){
float temp = ccs.calculateTemperature();
if(!ccs.readData()){
  ccs_msg.data = temp;
  chatter2.publish( &ccs_msg );
  
Serial.print("CO2: ");
Serial.print(ccs.geteCO2());
Serial.print("ppm, TVOC: ");
Serial.print(ccs.getTVOC());
Serial.print("ppb Temp:");
Serial.println(temp);

}
else{
Serial.println("ERROR!");
while(1);
  }
    }

     //bme bme 

     if (! bme.performReading()) {
    Serial.println("Failed to perform reading :(");
    return;
  }
  Serial.print("Temperature = ");
  Serial.print(bme.temperature);
  Serial.println(" *C");

  Serial.print("Pressure = ");
  Serial.print(bme.pressure / 100.0);
  Serial.println(" hPa");

  Serial.print("Humidity = ");
  Serial.print(bme.humidity);
  Serial.println(" %");

  Serial.print("Gas = ");
  Serial.print(bme.gas_resistance / 1000.0);
  Serial.println(" KOhms");

  Serial.print("Approx. Altitude = ");
  Serial.print(bme.readAltitude(SEALEVELPRESSURE_HPA));
  Serial.println(" m");

  Serial.println();
  chatter3.publish(&bme_msg);

  //spectro spectro 

  sensor.takeMeasurements(); //This is a hard wait while all 18 channels are measured

  Serial.print(sensor.getCalibratedA()); //410nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedB()); //435nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedC()); //460nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedD()); //485nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedE()); //510nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedF()); //535nm
  Serial.print(",");

  Serial.print(sensor.getCalibratedG()); //560nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedH()); //585nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedR()); //610nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedI()); //645nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedS()); //680nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedJ()); //705nm
  Serial.print(",");

  Serial.print(sensor.getCalibratedT()); //730nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedU()); //760nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedV()); //810nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedW()); //860nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedK()); //900nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedL()); //940nm
  Serial.print(",");

  Serial.println();
   chatter4.publish(&spectro_msg);

   //mq2 mq2 

   float mq2=analogRead(pin_mq2);
  /*
    Exponential regression:
    Gas    | a      | b
    H2     | 987.99 | -2.162
    LPG    | 574.25 | -2.222
    CO     | 36974  | -3.109
    Alcohol| 3616.1 | -2.675
    Propane| 658.71 | -2.168
  */
  //_PPM =  a*ratio^b
   mq2= (mq2 * (5.0 / 1023.0));
   float mq2_rs = ((5.0 * 10.0) / mq2) - 10.0;
   float ratio_mq2 = mq2_rs/RO;
   float LPG = 574.25*pow(ratio_mq2,-2.222);
   float alcohol = 3616.1*pow(ratio_mq2,-2.675);
   float propanol = 658.71*pow(ratio_mq2,-2.168);
mq2_msg.x = LPG;
  mq2_msg.y = alcohol;
  mq2_msg.z = propanol;
  chatter5.publish(&mq2_msg);

  //mq131 mq131 

  MQ131.update(); // Update data, the arduino will be read the voltage on the analog pin
  MQ131.readSensor(); // Sensor will read PPM concentration using the model and a and b values setted before or in the setup
  MQ131.serialDebug(); // Will print the table on the serial port
   chatter6.publish(&mq131_msg);

   //o2 o2 

   float Vout =0;
    Serial.print("Vout =");
 
    Vout = readO2Vout();
    Serial.print(Vout);
    Serial.print(" V, Concentration of O2 is ");
    Serial.println(readConcentration());
    delay(500);
}
 
float readO2Vout()
{
    long sum = 0;
    for(int i=0; i<32; i++)
    {
        sum += analogRead(pinAdc);
    }
 
    sum >>= 5;
 
    float MeasuredVout = sum * (VRefer / 1023.0);
    return MeasuredVout;
}
 
float readConcentration()
{
    // Vout samples are with reference to 3.3V
    float MeasuredVout = readO2Vout();
 
    //float Concentration = FmultiMap(MeasuredVout, VoutArray,O2ConArray, 6);
    //when its output voltage is 2.0V,
    float Concentration = MeasuredVout * 0.21 / 2.0;
    float Concentration_Percentage=Concentration*100;
    return Concentration_Percentage;
    chatter7.publish(&O2_msg);

    //ph ph 

    for(int i=0;i<10;i++)       //Get 10 sample value from the sensor for smooth the value
  { 
    buf[i]=analogRead(SensorPin);
    delay(10);
  }
  for(int i=0;i<9;i++)        //sort the analog from small to large
  {
    for(int j=i+1;j<10;j++)
    {
      if(buf[i]>buf[j])
      {
        temp=buf[i];
        buf[i]=buf[j];
        buf[j]=temp;
      }
    }
  }
  avgValue=0;
  for(int i=2;i<8;i++)                      //take the average value of 6 center sample
    avgValue+=buf[i];
  float phValue=(float)avgValue*5.0/1024/6; //convert the analog into millivolt
  phValue=3.5*phValue;                      //convert the millivolt into pH value
 /* Serial.print("    pH:");  
  Serial.print(phValue,2);
  Serial.println(" ");
  digitalWrite(13, HIGH);       
  delay(800);
  digitalWrite(13, LOW); */
 chatter8.publish(&ph_msg);

 //uv uv

float sensorVoltage; 
  float sensorValue;
 
  sensorValue = analogRead(A0);
  sensorVoltage = sensorValue/1024*3.3;
  Serial.print("sensor reading = ");
  Serial.print(sensorValue);
  Serial.println("");
  Serial.print("sensor voltage = ");
  Serial.print(sensorVoltage);
  Serial.println(" V");
  chatter9.publish(&uv_msg);
  nh.spinOnce();
  delay(500);

  }
  

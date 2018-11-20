import RPi.GPIO as GPIO
from time import sleep

#set up servo code
def SetupServo():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(3, GPIO.OUT)
    pwm=GPIO.PWM(3,50)
    pwm.start(0)
    return pwm

#function for setting angle on servo
def SetAngle(pwm, angle):
    print(angle)
    duty = angle / 18 + 2
    GPIO.output(3, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(3,False)
    pwm.ChangeDutyCycle(0)

def A_Press(pwm):
    SetAngle(pwm, 115)
    SetAngle(pwm, 100)

def L_Press(pwm):
    SetAngle(pwm, 5)
    SetAngle(pwm, 40)

def R_Press(pwm):
    SetAngle(pwm, 80)
    SetAngle(pwm, 40)

def ServoEnd(pwm):
    pwm.stop()
    GPIO.cleanup()

#pwm = SetupServo()
#A_Press(pwm)


#required cleanup code
#pwm.stop()
#GPIO.cleanup()

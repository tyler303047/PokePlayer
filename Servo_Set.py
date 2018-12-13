import RPi.GPIO as GPIO
from time import sleep

#set up servo code
def SetupServo():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(2, GPIO.OUT)
    GPIO.setup(3, GPIO.OUT)
    GPIO.setup(4, GPIO.OUT)
    GPIO.setup(14, GPIO.OUT)
    GPIO.setup(15, GPIO.OUT)
    pwm = []
    pwm.append(GPIO.PWM(2,50))
    pwm.append(GPIO.PWM(3,50))
    pwm.append(GPIO.PWM(4,50))
    pwm.append(GPIO.PWM(14,50))
    pwm.append(GPIO.PWM(15,50))
##    pwm.start(0)
    for p in pwm:
        p.start(0)
    return pwm

#function for setting angle on servo
def SetAngle(pwm, angle, pin):
    print(angle)
    duty = angle / 18 + 2
    GPIO.output(pin, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(pin,False)
    pwm.ChangeDutyCycle(0)

def A_Press(pwm):
    SetAngle(pwm[0], 90, 2)
    SetAngle(pwm[0], 130, 2)
    SetAngle(pwm[0], 90, 2)

def L_Press(pwm):
    SetAngle(pwm[1], 170, 3)
    SetAngle(pwm[1], 100, 3)

def R_Press(pwm):
    SetAngle(pwm[2], 120, 4)
    SetAngle(pwm[2], 80, 4)

def D_Press(pwm):
    SetAngle(pwm[3], 50, 14)
    SetAngle(pwm[3], 100, 14)

def Screen_Press(pwm):
    SetAngle(pwm[4], 23, 15)
    SetAngle(pwm[4], 70, 15)

def ServoEnd(pwm):
##    pwm.stop()
    for p in pwm:
        p.stop()
    GPIO.cleanup()

#pwm = SetupServo()
#A_Press(pwm)


#required cleanup code
#pwm.stop()
#GPIO.cleanup()

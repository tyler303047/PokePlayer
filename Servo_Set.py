import RPi.GPIO as GPIO
from time import sleep

#set up servo code
def SetupServo():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(3, GPIO.OUT)
    GPIO.setup(5, GPIO.OUT)
    GPIO.setup(7, GPIO.OUT)
    GPIO.setup(8, GPIO.OUT)
    pwm = []
    pwm.append(GPIO.PWM(3,50))
    pwm.append(GPIO.PWM(5,50))
    pwm.append(GPIO.PWM(7,50))
    pwm.append(GPIO.PWM(8,50))
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
    SetAngle(pwm[0], 115, 3)
    SetAngle(pwm[0], 100, 3)

def L_Press(pwm):
    SetAngle(pwm[1], 5, 5)
    SetAngle(pwm[1], 40, 5)

def R_Press(pwm):
    SetAngle(pwm[2], 80, 7)
    SetAngle(pwm[2], 40, 7)

def D_Press(pwm):
    SetAngle(pwm[3], 80, 8)
    SetAngle(pwm[3], 40, 8)

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

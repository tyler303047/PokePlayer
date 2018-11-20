#imports
from Servo_Control import *
import tensorflow
import cv2
import time

SLEEP_LENGTH_1 = 1
state_now = 'Encounter'

#functions
def Find_Pokemon:
    

#State Machine Functions
def Encounter:
    while(True):
        L_Button()
        if(Find_Pokemon()):
            return
        R_Button()
        if(Find_Pokemon()):
            return
def Run:
    #TODO: Add button presses I need

def Battle:
    while(True):
        A_Button()
        time.sleep(SLEEP_LENGTH_1)
        A_Button()
        time.sleep(SLEEP_LENGTH_1)
        if not Found_Pokemon():
            return

def Capture:
    D_Button()
    time.sleep(SLEEP_LENGTH_1)
    A_Button()
    time.sleep(SLEEP_LENGTH_1)

#main
def main:
    #start stage
    states = {
        'Encounter': Encounter,
        'Run': Run,
        'Battle': Battle,
        'Capture': Capture
        }

    #
    while(True):
        states[state_now]()
        
if __name__ == "__main__":
    main()


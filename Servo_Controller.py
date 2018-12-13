#imports
from Servo_Set import *
import cv2
import time
from obj_dect_lib import *
import RPi.GPIO as G
from Disp_Lib import *

Patrat = True
Battle = True
SLEEP_LENGTH_1 = 1
SLEEP_LENGTH_2 = 3
SLEEP_LENGTH_3 = 5
PATRAT_PIN = 26
BATTLE_PIN = 21
G.setmode(G.BCM)
G.setup(PATRAT_PIN, G.IN)
G.setup(BATTLE_PIN, G.IN)
state_now = 'Encounter'
#LCD_init()
pwm = SetupServo()
CONFIDENCE_IN_REASON, CLASSES, label_map, categories, category_index, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, detection_graph, sess, video = NN_Init()

#functions
def Find_Pokemon():
    #TODO: add code for object detection neural network
    return Obj_Dect(CONFIDENCE_IN_REASON, CLASSES, label_map, categories, category_index, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, detection_graph, sess, video)

#State Machine Functions
def Encounter():
    L_Press(pwm)
    R_Press(pwm)
    time.sleep(SLEEP_LENGTH_3)
    find, what = Find_Pokemon()
    if(find):
        print('found pokemon')
        global state_now
        global Battle
        state_now = 'Battle'
        #print('Patrat: ', Patrat, ' what: ', what, 'Battle: ', Battle)
        if (Patrat and what == 'Patrat') or ((not Patrat) and (what == 'Lillipup')):
            print('Patrat: ', Patrat, ' ', not Patrat, 'what == \'Lillipup\' ', what == 'Lillipup', ' Both: ', (not Patrat) and (what == 'Lillipup')) 
            if Battle:
                state_now = 'Battle'
            else:
                state_now = 'Capture'
        else:
            state_now = 'Run'
        print('state_now: ', state_now)
        return
    
def Run():
    D_Press(pwm)
    D_Press(pwm)
    L_Press(pwm)
    L_Press(pwm)
    R_Press(pwm)
    A_Press(pwm)
    time.sleep(SLEEP_LENGTH_2)
    global state_now
    state_now = 'Encounter'
    return

def Battle():
    Screen_Press(pwm)
    L_Press(pwm)
    A_Press(pwm)
    A_Press(pwm)
    time.sleep(SLEEP_LENGTH_3)
    find, what = Find_Pokemon()
    if not (find):
        print('pokemon gone')
        A_Press(pwm)
        time.sleep(SLEEP_LENGTH_1)
        global state_now
        state_now = 'Encounter'
        return

def Capture():
    L_Press(pwm)
    L_Press(pwm)
    A_Press(pwm)
    R_Press(pwm)
    A_Press(pwm)
    A_Press(pwm)
    A_Press(pwm)
    time.sleep(SLEEP_LENGTH_3)
    find, what = Find_Pokemon()
    if not (find):
        print('pokemon gone')
        A_Press(pwm)
        D_Press(pwm)
        A_Press(pwm)
        time.sleep(SLEEP_LENGTH_1)
        global state_now
        state_now = 'Encounter'
        return

def whichSwitch():
    global Patrat
    global Battle
    if G.input(PATRAT_PIN):
        Patrat = True
    else:
        Patrat = False
    if G.input(BATTLE_PIN):
        Battle = True
    else:
        Battle = False
    print('Patrat: ', Patrat)
    print('Battle: ', Battle)
    return

def Display():
    global state_now
    global Patrat
    global Battle
    LCD_Draw(Patrat, Battle, state_now)
    return

#main
def main():
    #start stage
    states = {
        'Encounter': Encounter,
        'Run': Run,
        'Battle': Battle,
        'Capture': Capture
        }
##    states = {
##        'Encounter': Encounter,
##        }
    #run functions
    while(True):
        print('state_now ', state_now)
        whichSwitch()
        Display()
        states[state_now]()
        
if __name__ == "__main__":
    main()


#imports
from Servo_Set import *
import cv2
import time
from obj_dect_lib import *

SLEEP_LENGTH_1 = 1
state_now = 'Encounter'
pwm = SetupServo()
CONFIDENCE_IN_REASON, CLASSES, label_map, categories, category_index, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, detection_graph, sess, video = NN_Init()

#functions
def Find_Pokemon():
    #TODO: add code for object detection neural network
    return Obj_Dect(CONFIDENCE_IN_REASON, CLASSES, label_map, categories, category_index, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, detection_graph, sess, video)

#State Machine Functions
def Encounter():
    while(True):
        L_Press(pwm)
        if(Find_Pokemon()):
            print('found pokemon')
            global state_now
            state_now = 'Battle'
            print('state_now? ', state_now)
            return
        R_Press(pwm)
        if(Find_Pokemon()):
            print('found pokemon')
            global state_now
            print('state_now? ', state_now)
            state_now = 'Battle'
            print('state_now? ', state_now)
            return
def Run():
    #TODO: Add button presses I need
    state_now = 'Encounter'
    return

def Battle():
    while(True):
        A_Press(pwm)
        time.sleep(SLEEP_LENGTH_1)
        A_Press(pwm)
        time.sleep(SLEEP_LENGTH_1)
        if not Find_Pokemon():
            print('pokemon gone')
            global state_now
            state_now = 'Encounter'
            return

def Capture():
    D_Press(pwm)
    time.sleep(SLEEP_LENGTH_1)
    A_Press(pwm)
    time.sleep(SLEEP_LENGTH_1)

#main
def main():
    #start stage
    states = {
        'Encounter': Encounter,
        'Run': Run,
        'Battle': Battle,
        'Capture': Capture
        }

    #
    while(True):
        print('state_now ', state_now)
        states[state_now]()
        
if __name__ == "__main__":
    main()


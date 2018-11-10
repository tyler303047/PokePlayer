#import cv2
from cv2 import *
import time
import numpy as np
import tensorflow as tf


CATEGORIES = ["lil","none","pat"]
model = tf.keras.models.load_model("64x5-CNN.model")

def prepare(frame):
    IMG_SIZE = 400
    #img_array = cv2.imread(filepath)
    new_color =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(new_color,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

cap = VideoCapture(0)

while(True):
    
    time.sleep(.05)

    ret , frame = cap.read()

    imshow('frame', frame)

    prediction = model.predict([prepare(frame)])

    output_label = CATEGORIES[np.argmax(prediction)]
    print('pokemon: ', output_label)

    if waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
destroyAllWindows()

import os
import cv2
import numpy as np
import tensorflow as tf

CATEGORIES = ["lil","pat"]
CATEGORIES2 = ["lil", "none", "pat"]

model1 = "64x5-CNN.model"
model2 = "64x5-CNN_classify_2.model"

def prepare(filepath):
    IMG_SIZE = 400
    img_array = cv2.imread(filepath)
    new_color =  cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(new_color,(IMG_SIZE,IMG_SIZE))
    new_array = new_array / 255
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


model = tf.keras.models.load_model(model1)

for folderName, subfolders, filenames in os.walk('./testing_batch3'):
        for filename in filenames:
            prediction = model.predict([prepare(folderName + '\/' +  str(filename))])
            label_here = folderName.split("testing_batch3\\",1)
            output_label = CATEGORIES2[np.argmax(prediction)]
            print('pokemon: ', label_here,' : ', output_label)

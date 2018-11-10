from cv2 import *
import random
import os

num = 314
def prepare(filepath):
    IMG_SIZE = 400
    img_array = cv2.imread(filepath)
    new_color =  cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(new_color,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(IMG_SIZE, IMG_SIZE, 3)

images = []

for folderName, subfolders, filenames in os.walk('./Patrat'):
        for filename in filenames:
            images.append(prepare(folderName + '\/' +  str(filename)))

            

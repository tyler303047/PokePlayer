import tensorflow as tf
from tensorflow.keras import utils
from sklearn.preprocessing import LabelEncoder
from random import shuffle
from cv2 import *
import numpy as np
import os
import pickle
import time

IMG_SIZE = 250

def prepare(filepath):
    IMG_SIZE1 = IMG_SIZE
    img_array = cv2.imread(filepath)
    new_color =  cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(new_color,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(IMG_SIZE1, IMG_SIZE1, 3)

training = True
#training = False
saved=False
#saved=True
img_dir = './PokemonData_run_2'
model_dir = '64x5-CNN_classify_2.model'
images = []
encoder = LabelEncoder()
print('started')

for folderName, subfolders, filenames in os.walk(img_dir):
        for filename in filenames:
            label_here = folderName.split("PokemonData_run_2\\",1)[1]
            images.append((prepare(folderName + '\/' +  str(filename)), label_here))

#images = shuffle(images)
shuffle(images)
#print(images)
train_images = np.zeros((4602,IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
train_label_list = []
for i, tuple1 in enumerate(images):
    train_images[i, :, :, :] = tuple1[0]
    train_label_list.append(tuple1[1])

images1 = np.array(images)
#print(images1.shape)

encoder.fit(train_label_list)
train_labels = encoder.transform(train_label_list)
##print('len1: ', len(train_labels))
##print('len1: ', len(train_labels_1))
##print(train_labels_1)
##for i,x in enumerate(train_labels_1):
##    train_labels[i] = train_labels_1[i]
dense_layers = 2
dense_neurons = 128
neurons = 256
conv_layers = 8
sizep = 4
##train_images = train_images / 255
##for img in train_images:
##    img = img / 255
train_images = [x / 255.0 for x in train_images]



if saved:
##    model = tf.keras.models.load_model(model_dir)
    print('heh')
else:
    print('here')
    model = tf.keras.Sequential()
    print('here1')
    model.add(tf.keras.layers.Conv2D(neurons, kernel_size=(3,3), input_shape=(IMG_SIZE,IMG_SIZE,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(tf.keras.layers.Conv2D(neurons, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(tf.keras.layers.Conv2D(neurons, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(tf.keras.layers.Conv2D(neurons, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(tf.keras.layers.Conv2D(neurons, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(tf.keras.layers.Flatten())
    for i in range(dense_layers):
        model.add(tf.keras.layers.Dense(dense_neurons, activation=tf.nn.relu))
    #output
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.softmax))

print(train_labels)

#Finish step 2
if training:
    print('train')
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.fit(np.array(train_images), train_labels, batch_size=2, shuffle=True, epochs = 5, validation_split=0.3)
    model.save(model_dir)




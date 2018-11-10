import tensorflow as tf
from tensorflow.keras import utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from random import shuffle
import numpy as np
import os
import pickle
import time

#NAME = "Pokemon-cnn-64x2-{}".format(int(time.time()))

#tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME),write_graph=False, write_grads=True)

checkpoint_path = './model1.ckpt'
training = True
#training = False
#pickeled = True
pickeled = False
saved=False
#saved=True
#encoder = LabelBinarizer()
encoder = LabelEncoder()
IMG_SIZE = 400
tf.reset_default_graph()

def get_img_Data():
    tensors = []
    results = []
    labels = []
    done = False
    #done = True
    print('started')
    
    for folderName, subfolders, filenames in os.walk('./testing_batch3'):
                
        for filename in filenames:
            filename_queue = tf.train.string_input_producer([folderName + '/' + filename])
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            my_img = tf.image.decode_jpeg(value, channels=3)
            resized_image = tf.image.resize_images(my_img, [IMG_SIZE, IMG_SIZE])
            tensors = tensors + [resized_image]
            #tensors = tensors + [my_img]
            #print(folderName.split("./testing_batch3\\",1)[1])
            labels = labels + [folderName.split("./testing_batch3\\",1)[1]]

    print('got out')

    temp_both_array = []
    print('at temp arrays')
    for i,x in enumerate(tensors):
        temp_both_array.append([x, labels[i]])

    shuffle(temp_both_array)

    tensors = []
    labels = []
    for i in temp_both_array:
        tensors.append(i[0])
        labels.append(i[1])

    #print('labels: ', labels)

    sess = tf.Session()
    test_result = []
    print('at session')
    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        i = 0
        print('going into eval')
        for t in tensors:
            print(i)
            i = i+1
            result = t.eval()
            results = results + [result]
        print('outta eval')

        coord.request_stop()
        coord.join(threads)
    
    transfomed_train_label = encoder.fit_transform(labels)

    #train_images = results / 255
    train_images = []
    for i in results:
        train_images.append(i / 255)

    return train_images, transfomed_train_label

def getSomeImages():
    tensors = []
    results = []
    #done = True
    print('started getting some')

    for folderName, subfolders, filenames in os.walk('./testing'):
                
        for filename in filenames:
            filename_queue = tf.train.string_input_producer([folderName + '/' + filename])
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            my_img = tf.image.decode_jpeg(value, channels=3)
            resized_image = tf.image.resize_images(my_img, [IMG_SIZE, IMG_SIZE])
            tensors = tensors + [resized_image]
            #tensors = tensors + [my_img]
            #print(folderName.split("Pokemon\\",1)[1])

    print('got out')

    sess = tf.Session()
    print('at session')
    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        i = 0
        print('going into eval')
        for t in tensors:
            i = i+1
            result = t.eval()
            results = results + [result]
            
        print('outta eval')

        coord.request_stop()
        coord.join(threads)

    for i in results:
        i = i / 255
    

    return results


if saved:
    model = tf.keras.models.load_model('64x5-CNN_1.model')
else:
    model = tf.keras.Sequential()
    #1st convolution
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), input_shape=(400,400,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #2nd convolution
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #3rd convolution
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #4th convolution
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #5th convolution
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #6th convolution
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #output
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))



#Finish step 2
if training:
    #train_images, transfomed_train_label = get_img_Data()
    if not pickeled:
        print('here')
        train_images, transformed_train_label = get_img_Data()
        pickle_out = open("train_images.pickle", "wb")
        pickle.dump(train_images, pickle_out)
        pickle_out.close()
        pickle_out = open("train_labels.pickle", "wb")
        pickle.dump(transformed_train_label, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("train_images.pickle", "rb")
        train_images = pickle.load(pickle_in)
        pickle_in = open("train_labels.pickle", "rb")
        transformed_train_label = pickle.load(pickle_in)

    
    images1 = np.array(train_images)
    print('shape: ', images1.shape)
        
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.fit(train_images, transformed_train_label, batch_size=2, shuffle=True, epochs = 5, validation_split=0.3, steps_per_epoch=500)
    model.save('64x5-CNN_1.model')

#Test prediction on some test image
##results = getSomeImages()
##scores = []
##for i in results:
##    i = i.reshape(1, IMG_SIZE, IMG_SIZE, 3)
##    results.append(model.predict([i]))
##print('scores: ', scores)
##print('scores2: ', scores2)
##print('????: ', encoder.inverse_transform(scores))
##print('????: ', encoder.inverse_transform(scores2))
#print(np.argmax(scores))
    


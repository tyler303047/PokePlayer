import tensorflow as tf
import os

results = []
labels = []
done = False
#done = True
print('started')

# Construct a `Session` to execute the graph.
##sess = tf.Session()
##with sess.as_default():
##    coord = tf.train.Coordinator()
##    threads = tf.train.start_queue_runners(coord=coord)
    for folderName, subfolders, filenames in os.walk('./Pokemon'):
                
        for filename in filenames:
            filename_queue = tf.train.string_input_producer([folderName + '/' + filename])
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            my_img = tf.image.decode_jpeg(value, channels=1)
            resized_image = tf.image.resize_images(my_img, [299, 299])
            twodimg = tf.layers.Flatten()(resized_image)
            flatimg = tf.layers.Flatten()(twodimg)
            result = flatimg.eval()
            results = results + [result]
            
            print('result: ',result)
            print(folderName.split("Pokemon\\",1)[1])
            labels = labels + [folderName.split("Pokemon\\",1)[1]]

##    coord.request_stop()
##    coord.join(threads)

print(results)
train_images = results[:int(len(results) * .6)]
train_labels = labels[:int(len(labels) * .6)]
test_images = results[int(len(results) * .6):]
test_labels = labels[int(len(labels) * .6):]
print('len tensors: ', len(results))
print('len train: ', len(train_images))
print('len test: ', len(test_images))
print('len both together: ', len(train_images) + len(test_images))

#Step 2, Create neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(299,299)))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#Finish step 2
model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

#Train the model
model.fit(train_images, train_labels, epochs=5)

#Validate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print('Accuracy', test_accuracy)

#Test prediction on some test image
scores = model.predict(test_images[0:1])
print(np.argmax(scores))
    

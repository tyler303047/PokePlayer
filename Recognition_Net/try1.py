import tensorflow as tf

filename_queue = tf.train.string_input_producer(['C:\Python3\TensorflowExp\garbage_try1\filename0.jpg'])

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value, channels=1)
print('output 1: ', my_img)
resized_image = tf.image.resize_images(my_img, [299, 299])
print('output 2: ', resized_image)

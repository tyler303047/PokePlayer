import tensorflow as tf

filename_queue = tf.train.string_input_producer(['C:/Python3/TensorflowExp/Pokemon_data/filename0.jpg'])
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
my_img = tf.image.decode_jpeg(value, channels=1)
resized_image = tf.image.resize_images(my_img, [3, 3])
##resized_image = tf.zeros("Squeeze:0", shape=(3, 3, 1), dtype=tf.float32)
##print(resized_image)

### Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)
###e = resized_image

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(value)

print('result: ', result)

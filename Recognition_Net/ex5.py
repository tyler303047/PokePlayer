import tensorflow as tf

result = 0

filename_queue = tf.train.string_input_producer(['C:\Python3\TensorflowExp\Pokemon_data\filename0.jpg'])
#filename_queue = 'C:\Python3\TensorflowExp\Pokemon_data\filename0.jpg'
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
my_img = tf.image.decode_jpeg(value, channels=1)
resized_image = tf.image.resize_images(my_img, [299, 299])
print(resized_image)

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
print('we got somewhere')
#print(resized_image.eval())
init = tf.global_variables_initializer()
sess.run(init)
queue = tf.train.start_queue_runners(sess=sess)
print('1')
#tf.train.start_queue_all_variables()
# We can also use 'c.eval()' here.
with sess.as_default():
    print('2')
    print(resized_image.eval())
    print('3')

#sess.close()

print('result: ', result)

import tensorflow as tf

result = 0

#Get image into appropriately sized tensor
filename_queue = tf.train.string_input_producer([r'C:\Python3\TensorflowExp\Pokemon_data\filename0.jpg'])
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
my_img = tf.image.decode_jpeg(value, channels=1)
resized_image = tf.image.resize_images(my_img, [299, 299] )
twodimg = tf.layers.Flatten()(resized_image)
flatimg = tf.layers.Flatten()(twodimg)

# Construct a `Session` to execute the graph.
sess = tf.Session()

#Print out numpy array in tensor 
with sess.as_default():
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    result = flatimg.eval()

    coord.request_stop()
    coord.join(threads)
    print(result)

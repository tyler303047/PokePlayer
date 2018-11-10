import os
import cv2
import numpy as np
import sys
import time
from utils import label_map_util
from utils import visualization_utils as vis_util

CONFIDENCE_IN_REASON = .85

MODEL = 'inference_graph'

THIS_PATH = os.getcwd()

CKPT_PATH = os.path.join(THIS_PATH, MODEL,'frozen_inference_graph.pb')

LABEL_PATH = os.path.join(THIS_PATH,'training','labelmap.pbtxt')

CLASSES = 2

#Load label map
label_map = label_map_util.load_labelmap(LABEL_PATH)

#Load Categories
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

#Index categories
category_index = label_map_util.create_category_index(categories)

#Get tensors

#Input tensor is image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Load boxes(even though 
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Get the confidence scores and classes
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load Tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Initialize webcam video capture
video = cv2.VideoCapture(0)

while(True):

    ret, frame = video.read()

    # Perform the actual detection by running the model with the image as input
    (found_box, scores_list, classes_list, num_list) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame})
    
    #List pokemon with reasonable confidence
    print('-----')
    for i, x in enumerate(scores_list[0]):
        if x > CONFIDENCE_IN_REASON:
            output_guess = category_index[classes_list[0][i]]['name']
            print('found pokemon: ', output_guess)
    print('-----')
    
    # Press 'q' to quit or use ctrl+C
    if cv2.waitKey(1) == ord('q'):
        break

    #delay of half a second
    time.sleep(.5)

# Clean up
video.release()
cv2.destroyAllWindows()

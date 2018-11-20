import os
import cv2
import numpy as np
import sys
import time
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

def NN_Init():
    CONFIDENCE_IN_REASON = .85

    MODEL = 'inference_graph'

    THIS_PATH = os.getcwd()

    CKPT_PATH = os.path.join(THIS_PATH, MODEL,'frozen_inference_graph.pb')

    LABEL_PATH = os.path.join(THIS_PATH,'training','labelmap.pbtxt')

    CLASSES = 2

    #Load label map
    label_map = label_map_util.load_labelmap(LABEL_PATH)

    #Load Categories
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=CLASSES, use_display_name=True)

    #Index categories
    category_index = label_map_util.create_category_index(categories)

    #Get tensors
     # Load Tensorflow model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(CKPT_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    #Input tensor is image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Load boxes(even though 
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Get the confidence scores and classes
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam video capture
    video = cv2.VideoCapture(0)
    return CONFIDENCE_IN_REASON, CLASSES, label_map, categories, category_index, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, detection_graph, sess, video

##def Obj_Dect(video, detection_boxes, detection_scores, detection_classes, num_detections, sess):
def Obj_Dect(CONFIDENCE_IN_REASON, CLASSES, label_map, categories, category_index, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, detection_graph, sess, video):

    ret, frame = video.read()
    frame_ex = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (found_box, scores_list, classes_list, num_list) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_ex})
    
    #List pokemon with reasonable confidence
    print('-----')
    for i, x in enumerate(scores_list[0]):
        if x > CONFIDENCE_IN_REASON:
            output_guess = category_index[classes_list[0][i]]['name']
            print('found pokemon: ', output_guess)
            return True
    print('did not find pokemon')
    return False

# Clean up
def Obj_Dect_Clean():
    video.release()
    cv2.destroyAllWindows()

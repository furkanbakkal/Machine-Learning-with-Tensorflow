#!/usr/bin/env python
# coding: utf-8

"""
Furkan BAKKAL

Squid Game Challenge - November Project - Lunizz

21/10/2021

Github Repository:
Mail:
Discord:

"""

import draw

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # close warnings

import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # close warnings

import cv2

import numpy as np
from PIL import Image

import time
from object_detection.utils import label_map_util


#path of squid game image
image_path = "test.jpg"

#path of our label map
label_path = "exported-models/my_mobilenet_model/saved_model/label_map.pbtxt"

#path of detection threshold
threshold_v = 0.35

#path of trained model (detector)
model_path = "exported-models/my_mobilenet_model/saved_model"

print('Loading model... ')
start_time = time.time()

#load model
detect_fn = tf.saved_model.load(model_path)

end_time = time.time()
elapsed_time = end_time - start_time
print('OK! Took ' + str(elapsed_time) + ' seconds ')

#load label map
category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

print("Running...")

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
height, width, _ = image.shape
image_expanded = np.expand_dims(image_rgb, axis=0)

#convert to tensor
input_tensor = tf.convert_to_tensor(image)

input_tensor = input_tensor[tf.newaxis, ...]

detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
scores = detections['detection_scores']
boxes = detections['detection_boxes']
classes = detections['detection_classes']


for i in range(len(scores)):
    if ((scores[i] > threshold_v) and (scores[i] <= 1.0)):
 
        # Get bounding box coordinates and draw box

        ymin = int(max(1,(boxes[i][0] * height)))
        xmin = int(max(1,(boxes[i][1] * width)))
        ymax = int(min(height,(boxes[i][2] * height)))
        xmax = int(min(width,(boxes[i][3] * width)))
        
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 3)
        # Draw label
        object_name = category_index[int(classes[i])]['name'] #class index -> label name
        label =object_name # name it

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2) # Get font size

        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window

        cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box for writing label
        cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Draw label text

        if label=="umbrella":
            umb_x_min=xmin
            umb_y_min=ymin
            umbrella=True
            
    
        if label=="circle":
            circ_x_min=xmin
            circ_y_min=ymin
            circle=True
           
        if label=="triangle":
            tri_x_min=xmin
            tri_y_min=ymin
            triangle=True

        if label=="star":
            star_x_min=xmin
            star_y_min=ymin
            star=True         

print('Done')

if umbrella:
    draw.boxing(umb_x_min,umb_y_min,100,"umbrella")

if circle:
    draw.boxing(circ_x_min,circ_y_min,100,"circle")

if triangle:
    draw.boxing(tri_x_min,tri_y_min,100,"triangle")

if star:
    draw.boxing(star_x_min,star_y_min,100,"star")

#display output image
cv2.imshow('Squid Game Challange - Tensorflow Output', image)

#close window when ESC pressed
cv2.waitKey(0)

#cleanup
cv2.destroyAllWindows()


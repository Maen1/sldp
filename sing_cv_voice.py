
import os
import sys
import cv2
import imutils
import tarfile
import pyttsx3
import zipfile
import numpy as np
from PIL import Image
from io import StringIO
import tensorflow as tf
import six.moves.urllib as urllib
from collections import defaultdict
from matplotlib import pyplot as plt
import subprocess
from os import system
sys.path.append("..")

# object detection api helpers
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'tf-data/ssd-graph'
MODEL_FILE = MODEL_NAME + '.tar.gz'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('tf-data/training', 'object-detection.pbtxt')

NUM_CLASSES = 29

#load frozen tf graph
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

#labels
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

alphabet = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H',
			9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O',
			16:'P', 17:'Q', 18:'R', 19:'S',20:'T', 21:'U',22:'V', 23:'W',
			24:'X',25:'Y', 26:'Z', 27:'space', 28:'delete', 29:'empty'}

cap = cv2.VideoCapture(0)

# Running the tensorflow session
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		ret = True
		while (ret):
			ret,image_np = cap.read()
			image_np = cv2.flip(image_np,1)
			image_np =imutils.resize(image_np,width=800)
      #image_np = cv2.resize(image_np, (220, 220))
			if ret:
				x1, y1, x2, y2 = 100, 100, 300, 300
				img_cropped = image_np[y1:y2, x1:x2]
      
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(img_cropped, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# Actual detection.
			(boxes, scores, classes, num_detections) = sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
			vis_util.visualize_boxes_and_labels_on_image_array(
				img_cropped,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				min_score_thresh=.75,
				use_normalized_coordinates=True,
				line_thickness=6)
			print(np.squeeze(classes).astype(np.int32))
			print(np.squeeze(scores)[0])
			print(alphabet[np.squeeze(classes)[0].astype(np.int32)])
			if np.squeeze(scores)[0] > 0.85 and np.squeeze(classes)[0].astype(np.int32) < 27 :
				system('say ' +alphabet[np.squeeze(classes)[0].astype(np.int32)])
			cv2.rectangle(image_np, (x1, y1), (x2, y2), (255,0,0), 2)
			cv2.imshow('image',image_np)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				cap.release()
				break
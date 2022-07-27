import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from core.yolov4 import filter_boxes

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
######################################################################
# Utils for downloading and extracting zip files
# ----------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
# ######################################################################
# # Load pretrained TFLite model
# # ----------------------------

# Now we can open yolov3-416-int8_tf2_3_0.tflite
tflite_model_file = "yolov3-416-int8_tf2_3_0.tflite"
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Load a test image
# -----------------
input_size = 416
import cv2
import glob
img_list = []
org_list = []
for img in glob.glob("yolo_images/*.jpg"):

    original_image = cv2.imread(img)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    org_list.append(original_image)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    image_data = np.expand_dims(image_data, axis=0)
    image_data = np.asarray(image_data).astype(np.float32)


    img_list.append(image_data)




"""
file = "kite.jpg"

dir = os.getcwd()
img_path = os.path.join(dir, file)
"""

# Compile the model with relay
# ----------------------------

# TFLite input tensor name, shape and type
input_tensor = "input_1"
input_shape = (1, input_size, input_size, 3)
input_dtype = "float32"

# Parse TFLite model and convert it to a Relay module
from tvm import relay, transform

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

# Build the module against to x86 CPU
target = "llvm"
with transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)


import tvm
from tvm import te
from tvm.contrib import graph_executor as runtime
# Create a runtime executor module
module = runtime.GraphModule(lib["default"](tvm.cpu()))
# Feed input data
i=0
for i in range(50):
	module.set_input(input_tensor, tvm.nd.array(img_list[i]))

	module.run()

	pred0 = module.get_output(0).numpy()
	pred1 = module.get_output(1).numpy()

	boxes, pred_conf = filter_boxes(pred0, pred1, score_threshold=0.25,
									input_shape=tf.constant([input_size, input_size]))
	boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
		boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
		scores=tf.reshape(
			pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
		max_output_size_per_class=50,
		max_total_size=50,
		iou_threshold=0.45,
		score_threshold=0.25
	)
	pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
	images = utils.draw_bbox(org_list[i], pred_bbox)
	
	images = images.astype(np.float32)
	images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

	path = '/home/hsiancheng/Resnet50/yolo_results'

	name = str(i)+".jpg"
	cv2.imwrite(name, images)


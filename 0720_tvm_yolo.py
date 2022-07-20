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

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore
######################################################################
# Utils for downloading and extracting zip files
# ----------------------------------------------
import os
import numpy as np
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

file = "kite.jpg"
input_size = 416
dir = os.getcwd()
img_path = os.path.join(dir, file)

img = image.load_img(img_path, target_size=(input_size, input_size))
original_data = image.img_to_array(img)
original_data = original_data.astype(int)
image_data = np.expand_dims(original_data, axis=0)
image_data = preprocess_input(image_data)

######################################################################
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

# # # config = ConfigProto()
# # # config.gpu_options.allow_growth = True
# # session = InteractiveSession(config=config)
# # original_image = cv2.imread("kite.jpg")
# # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# # image_data = cv2.resize(original_image, (input_size, input_size))
# # image_data = image_data / 255.

# # interpreter = tf.lite.Interpreter(model_path=tflite_model)
# # interpreter.allocate_tensors()
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()
# # interpreter.set_tensor(input_details[0]['index'], images_data)
# # interpreter.invoke()

import tvm
from tvm import te
from tvm.contrib import graph_executor as runtime
# Create a runtime executor module
module = runtime.GraphModule(lib["default"](tvm.cpu()))
# Feed input data
module.set_input(input_tensor, tvm.nd.array(image_data))
# Run
# t0 = time.time()
module.run()
# print("tvm inference cost: {}".format(time.time() - t0))
# Get output
tvm_output = module.get_output(0).numpy()

# pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
pred0 = tvm_output
pred1 = np.zeros(tvm_output.shape, dtype=np.float32)
print(tvm_output.shape)
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
images = utils.draw_bbox(original_data, pred_bbox)
images = Image.fromarray(images.astype(np.float32))
images.show()
images = cv2.cvtColor(np.array(images), cv2.COLOR_BGR2RGB)
cv2.imwrite("kite_yolov3_result.jpg", images)

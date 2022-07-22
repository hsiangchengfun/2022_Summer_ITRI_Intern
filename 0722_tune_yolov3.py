import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from core.yolov4 import filter_boxes
import tvm
from tvm import te
from tvm.contrib import graph_executor as runtime
from tvm.contrib import graph_executor
import tflite
from tvm import relay, transform, autotvm
import os
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime
import time
from scipy.special import softmax
#input- input_1
#output- Identity
image_path = "kite.jpg"
input_size = 416
tflite_model = "yolov3-416-int8_tf2_3_0.tflite"


tflite_model_buf = open(tflite_model, "rb").read()


original_image = cv2.imread(image_path)



#trans to rgb type
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)



#size = 416
image_data = cv2.resize(original_image, (input_size, input_size))
image_data = image_data / 255.


# Add a dimension to the image so that we have NHWC format layout
image_data = np.expand_dims(image_data, axis=0)

#store image
images_data = []

input_tensor = "input_1"
input_shape = (1, 416, 416, 3)
input_dtype = "float32"

tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

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


dev = tvm.device(str(target), 0)



"""
images_data.append(image_data)
#transform image to array


images_data = np.asarray(images_data).astype(np.float32)



module = runtime.GraphModule(lib["default"](tvm.cpu()))

"""

module = runtime.GraphModule(lib["default"](dev))
module.set_input(input_tensor, image_data)
module.run()


unoptimized = []








import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
	np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
	* 1000
	/ timing_number
)
unoptimized = {
	"mean": np.mean(unoptimized),
	"median": np.median(unoptimized),
	"std": np.std(unoptimized),
}

print(unoptimized)

"""



module.set_input(input_tensor, tvm.nd.array(images_data))


t0 = time.time()
module.run()
cost = time.time() - t0

unoptimized.append(cost*1000)
output_shape = (1, 1000)

tvm_output = module.get_output(0).numpy()


scores = softmax(tvm_output)
scores = np.squeeze(scores)
pred = np.argmax(scores)

unoptimized = np.asarray(unoptimized)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}
print(unoptimized)  
"""





number = 10
repeat = 10
min_repeat_ms = 0
timeout = 10

runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)
tuning_option = {
    "tuner": "xgb",
    "trials": 1500,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "yolov3-autotuning.json",
}





# begin by extracting the tasks from the tflite model
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
	prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
	tuner_obj = XGBTuner(task, loss_type="rank")
	tuner_obj.tune(
		n_trial=min(tuning_option["trials"], len(task.config_space)),
		early_stopping=tuning_option["early_stopping"],
		measure_option=tuning_option["measure_option"],
		callbacks=[
			autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
			autotvm.callback.log_to_file(tuning_option["tuning_records"]),
		],
	)
	print(autotvm.callback)

with autotvm.apply_history_best(tuning_option["tuning_records"]):
	with tvm.transform.PassContext(opt_level=3, config={}):
		lib = relay.build(mod, target=target, params=params)

module = graph_executor.GraphModule(lib["default"](dev))

################################################################################
# Verify that the optimized model runs and produces the same results:

module.set_input(input_tensor, image_data)
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
images = utils.draw_bbox(original_image, pred_bbox)
images = images.astype(np.float32)
images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
cv2.imwrite("kite_yolov3_result.jpg", images)

timing_number = 10
timing_repeat = 10
optimized = (
	np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
	* 1000
	/ timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))

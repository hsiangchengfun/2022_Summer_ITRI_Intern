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

#input- input_1
#output- Identity
image_path = "kite.jpg"
input_size = 416
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
# A single cat dominates the examples!
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

#image_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
#image_path = download_testdata(image_url, "cat.png", module="data")
image_path = "kite.jpg"
#resized_image = Image.open(image_path).resize((224, 224))
resized_image = Image.open(image_path).resize((416, 416))
plt.imshow(resized_image)
plt.show()
image_data = np.asarray(resized_image).astype("float32")

# Add a dimension to the image so that we have NHWC format layout
#image_data = np.expand_dims(image_data, axis=0)

# Preprocess image as described here:
# https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
"""
image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
"""
image_data /= 255
print("input", image_data.shape)











if __name__ == '__main__':


	
	
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print(interpreter)

    
    
    input_details = interpreter.get_input_details()
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print(input_details)

    output_details = interpreter.get_output_details()
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print(output_details)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]


    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
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
    
    cv_v_image = cv2.cvtColor(np.asarray(resized_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('opencv image', cv_v_image)
#    cv2.waitKey(1)
    
    print(type(resized_image))
    print(type(cv_v_image))
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(cv_v_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite("kite_yolov3_result.jpg", image)



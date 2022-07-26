from tensorflow.keras.applications.resnet50 import ResNet50
import time
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
import os

fp=open("result_new.txt","w")


from scipy.special import softmax

model_dir = '/home/hsiancheng/ResNet50'

tflite_model_file = os.path.join(model_dir,"resnet50_uint8_tf2.1_20200911_quant.tflite")
tflite_model_buf = open(tflite_model_file,"rb").read()


# Get TFLite model from buffer
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)



# 預先訓練好的模型 -- ResNet50


model = ResNet50(weights='imagenet')

fipt = open("label.txt","r")
#origin = fipt.readlines()

image_list=[]
accu = 0
for i in range(100):
    
    origin = fipt.readline().splitlines()
    for j in range(10):
        if(i == 99 and j == 9): break
#        origin = origin  
        img_path = '/home/hsiancheng/ResNet50/image_classification/'+str(i)+'-'+str(j)+'.jpeg'

# 任意一張圖片，例如大象

        #img_path = './ResNet50/kite.jpg'

# 載入圖檔，並縮放寬高為 (224, 224) 

        img = image.load_img(img_path, target_size=(224, 224))

# 加一維，變成 (1, 224, 224, 3)，最後一維是色彩

        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

# 特徵縮放，每個特徵減掉該特徵的平均數
        #print(type(x))
        x = preprocess_input(x)
        #print(type(x))
        image_list.append(x)
# 預測

#        preds = model.predict(x)

# decode the results into a list of tuples (class, description, probability)
        
 #       origin = fipt.readline().splitlines()
#        print(origin[0])
 #       if(str(decode_predictions(preds, top=1)[0][0][0]) == str(origin[0])):
  #          accu = accu + 1
        #origin = str(origin.split(", ",1))

# 顯示預測前3名的答案
   #     fp.write(str(origin[0]) + "  " + str( decode_predictions(preds, top=1)[0][0][0])+"\n")
    #    print( origin[0] ,' Predicted:', decode_predictions(preds, top=1)[0][0][0])




input_tensor = "input_1"

input_shape = (1, 224, 224, 3)

input_dtype = "float32"






from tvm import relay, transform



mod, params = relay.frontend.from_tflite(

    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}

)




target = "llvm"

with transform.PassContext(opt_level=3):

    lib = relay.build(mod, target, params=params)







import tvm

from tvm import te

from tvm.contrib import graph_executor as runtime


dev = tvm.device(str(target), 0)


module = runtime.GraphModule(lib["default"](tvm.cpu()))

unoptimized=[]
total_opt=[]
for i in range(999):
    
    module.set_input(input_tensor, tvm.nd.array(image_list[i]))
    t0 = time.time()
    module.run()
    cost = time.time() - t0
    unoptimized.append(cost*1000)
    tvm_output = module.get_output(0).numpy()
    
    total_opt.append(tvm_output)
    #print("load img ",i,"\n")








################################################################################
# Collect Basic Performance Data
# ------------------------------
# We want to collect some basic performance data associated with this
# unoptimized model and compare it to a tuned model later. To help account for
# CPU noise, we run the computation in multiple batches in multiple
# repetitions, then gather some basis statistics on the mean, median, and
# standard deviation.
import timeit

timing_number = 10
timing_repeat = 10
unoptimized = np.asarray(unoptimized)
"""
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
#    / timing_number
)
"""
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)

################################################################################



"""
label_path = '/home/hsiancheng/label.txt'
pre_num = []

predictions=[]
prediction=[]
accu=0


AP = []
AR = []
TP = [] # predict it is and it is
FP = [] # predict it is  and it's not
FN = [] # predict it's not and it is
used_label = {}
Total_AP = 0
Total_AR = 0



for i in range (1000):
	FP.append(0)
	TP.append(0)
	FN.append(0)
	AP.append(0)
	AR.append(0)

with open(label_path) as f:

    labels = f.readlines()

for i in range(999):

    predictions.append( np.squeeze(total_opt[i]))

    prediction.append( np.argmax(predictions[i]))
    print("run predict ",i,"\n")

    print("origin: ",labels[int(i/10)]," predict: ",labels[prediction[i]])
    
    
    
    used_label[int(i/10)] = 1
    used_label[prediction[i]]=1
    if( labels[int(i/10)] == labels[prediction[i]] ):
        accu = accu +1
        TP[int(i/10)] += 1
    else:
    	FP[prediction[i]] += 1
    	FN[int(i/10)] += 1
   
   
   
   """

################################################################################
# Tune the model
# --------------
# The previous model was compiled to work on the TVM runtime, but did not
# include any platform specific optimization. In this section, we will show you
# how to build an optimized model using TVM to target your working platform.
#
# In some cases, we might not get the expected performance when running
# inferences using our compiled module. In cases like this, we can make use of
# the auto-tuner, to find a better configuration for our model and get a boost
# in performance. Tuning in TVM refers to the process by which a model is
# optimized to run faster on a given target. This differs from training or
# fine-tuning in that it does not affect the accuracy of the model, but only
# the runtime performance. As part of the tuning process, TVM will try running
# many different operator implementation variants to see which perform best.
# The results of these runs are stored in a tuning records file.
#
# In the simplest form, tuning requires you to provide three things:
#
# - the target specification of the device you intend to run this model on
# - the path to an output file in which the tuning records will be stored
# - a path to the model to be tuned.
#

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

################################################################################
# Set up some basic parameters for the runner. The runner takes compiled code
# that is generated with a specific set of parameters and measures the
# performance of it. ``number`` specifies the number of different
# configurations that we will test, while ``repeat`` specifies how many
# measurements we will take of each configuration. ``min_repeat_ms`` is a value
# that specifies how long need to run configuration test. If the number of
# repeats falls under this time, it will be increased. This option is necessary
# for accurate tuning on GPUs, and is not required for CPU tuning. Setting this
# value to 0 disables it. The ``timeout`` places an upper limit on how long to
# run training code for each tested configuration.

number = 10
repeat = 10
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 40  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

################################################################################
# Create a simple structure for holding tuning options. We use an XGBoost
# algorithim for guiding the search. For a production job, you will want to set
# the number of trials to be larger than the value of 20 used here. For CPU we
# recommend 1500, for GPU 3000-4000. The number of trials required can depend
# on the particular model and processor, so it's worth spending some time
# evaluating performance across a range of values to find the best balance
# between tuning time and model optimization. Because running tuning is time
# intensive we set number of trials to 10, but do not recommend a value this
# small. The ``early_stopping`` parameter is the minimum number of trails to
# run before a condition that stops the search early can be applied. The
# measure option indicates where trial code will be built, and where it will be
# run. In this case, we're using the ``LocalRunner`` we just created and a
# ``LocalBuilder``. The ``tuning_records`` option specifies a file to write
# the tuning data to.

tuning_option = {
    "tuner": "xgb",
    "trials": 1500,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "0722_resnet-50-autotuning.tfrecord",
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



with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = runtime.GraphModule(lib["default"](dev))

################################################################################
# Verify that the optimized model runs and produces the same results:

dtype = "float32"

scores=[]



optimized = []

total_opt=[]
for i in range(999):
    
    module.set_input(input_tensor, tvm.nd.array(image_list[i]))
    t0 = time.time()
    module.run()
    cost = time.time() - t0
    optimized.append(cost*1000)
    tvm_output = module.get_output(0).numpy()
    
    total_opt.append(tvm_output)
    #print("load img ",i,"\n")

    scores.append(softmax(tvm_output))
    scores[i] = np.squeeze(scores[i])
    ranks = np.argsort(scores[i])[::-1]




import timeit

timing_number = 10
timing_repeat = 10


optimized = np.asarray(optimized)
"""
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
"""
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))

 
   
   



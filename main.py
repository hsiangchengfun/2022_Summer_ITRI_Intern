from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
import os

fp=open("result_new.txt","w")



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





module = runtime.GraphModule(lib["default"](tvm.cpu()))


total_opt=[]
for i in range(999):
    
    module.set_input(input_tensor, tvm.nd.array(image_list[i]))
    
    module.run()

    tvm_output = module.get_output(0).numpy()
    
    total_opt.append(tvm_output)
    print("load img ",i,"\n")









label_path = '/home/hsiancheng/label.txt'
pre_num = []

predictions=[]
prediction=[]
accu=0

with open(label_path) as f:

    labels = f.readlines()

for i in range(999):

    predictions.append( np.squeeze(total_opt[i]))

    prediction.append( np.argmax(predictions[i]))
    print("run predict ",i,"\n")

    print("origin: ",labels[int(i/10)]," predict: ",labels[prediction[i]])

    if( labels[int(i/10)] == labels[prediction[i]] ):
        accu = accu +1




print("accuracy == ",(accu/999))


"""

print(origin[0])
#print(origin)
print(np.size(origin))
print(accu)
print("accuracy == ",accu/999)
"""


import os

fp = open ("result.txt","w")



def extract(path):

    import tarfile



    if path.endswith("tgz") or path.endswith("gz"):

        dir_path = os.path.dirname(path)

        tar = tarfile.open(path)

        tar.extractall(path=dir_path)

        tar.close()

    else:

        raise RuntimeError("Could not decompress the file: " + path)




model_dir = '/home/hsiancheng/ResNet50'

tflite_model_file = os.path.join(model_dir,"resnet50_uint8_tf2.1_20200911_quant.tflite")
tflite_model_buf = open(tflite_model_file,"rb").read()




try:

    import tflite



    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

except AttributeError:

    import tflite.Model



    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)




from PIL import Image

from matplotlib import pyplot as plt

import numpy as np



image_list=[]

for i in range(100):
    for j in range(10):
        if(i == 99 and j == 9): break   
        image_path = '/home/hsiancheng/ResNet50/image_classification/'+str(i)+'-'+str(j)+'.jpeg'

        resized_image = Image.open(image_path).resize((224, 224))
        image_data = np.asarray(resized_image).astype("float32")



        image_data = np.expand_dims(image_data, axis=0)


        image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1

        image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1

        image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1


        #print("input", image_data.shape)



        image_list.append(image_data)








"""
#image_path = '/home/hsiancheng/ResNet50/kite.jpg'



resized_image = Image.open(image_path).resize((224, 224))

    #plt.imshow(resized_image)

    #plt.show()

image_data = np.asarray(resized_image).astype("float32")



image_data = np.expand_dims(image_data, axis=0)


image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1

image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1

image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1


print("input", image_data.shape)



image_list.append(image_data)


"""




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









label_path = '/home/hsiancheng/ResNet50/imagenet1000_labels.txt'
pre_num = []

predictions=[]
prediction=[]
accu=0

with open(label_path) as f:

    labels = f.readlines()

for i in range(999):

    predictions.append( np.squeeze(total_opt[i]))

    prediction.append( np.argmax(predictions[i]))


    print("origin: ",int(i/10)," predict: ",labels[prediction[i]])

    if( int(i/10) == labels[prediction[i]].split(' ',1)[0] ):
        accu = accu +1




print("accuracy == ",float(accu/999))

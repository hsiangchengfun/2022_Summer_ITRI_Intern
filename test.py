import cv2
import glob
img_list = []
for img in glob.glob("yolo_images/*.jpg"):
    n= cv2.imread(img)
    img_list.append(n)
    
    
    
    
print(len(img_list))

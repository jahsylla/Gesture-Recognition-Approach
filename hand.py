
from __future__ import print_function
import os
from palm_detector import * 
import numpy as np
from PIL import Image
from tqdm import tqdm
   
palm_model_path = "./models/palm_detection_without_custom_op.tflite"
anchors_path = "./data/anchors.csv"


input_path = "F:/20bn-jester/20bn-jester-v1/"
output_path = "F:/20bn-jester/bbox_output/"



detector = PalmDetector(palm_model_path, anchors_path)



files = list(set(os.listdir(input_path)))

for fn in tqdm(files):
    if not os.path.exists(output_path + fn):
        os.makedirs(output_path + fn)
    list_image = os.listdir(input_path + fn)
    for name_img in tqdm(list_image):
        img = Image.open(input_path + fn + "/" + name_img)
        img = np.array(img)
        box_list = detector(img)
        print(output_path + fn + "/" + name_img.split(".")[0] + ".npy")
        np.save(output_path + fn + "/" + name_img.split(".")[0] + ".npy", box_list)


# path_im = "obama.jpg"
# img = Image.open(path_im)
# img = np.array(img)
# import matplotlib.pyplot as plt
# #plt.imshow(img)

# box_list = detector(img)
# print(box_list)


# img = Image.open(path_im)
# img2 = img.crop((min(box_list[:,0]), min(box_list[:,1]), max(box_list[:,0]), max(box_list[:,1])))
# plt.imshow(img2)
# plt.show() 


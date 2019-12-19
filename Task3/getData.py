import glob
import os
import numpy as np
import cv2
import natsort
from PIL import Image


#
# file_name ='face/s1/1.bmp'
# for name in glob.glob(file_name):
#     print('name',name)





def get_picture(all_data_set,all_data_label):
    folder = "face"
    path = folder
    files = os.listdir(path)
    files =  natsort.natsorted(files)
    print('files',files)
    print('files',len(files))
    for i in range(len(files)):
        label= i+1
        for j in range(1,11):
                file_name = folder + "/"+ files[i] + "/"+ str(j)+ ".bmp"
                print('file_name',file_name)
                for name in glob.glob(file_name):
                    img = Image.open(name)
                    all_data_set.append(list(img.getdata()))
                    all_data_label.append(label)
    return all_data_set,all_data_label



all_data_set = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = []  # 总数据对应的类标签
all_data_set, all_data_label = get_picture(all_data_set,all_data_label)
print('all_data_set',np.array(all_data_set).shape)
print('all_data_label',len(all_data_label))


# files =  natsort.natsorted(files)
#
# for i in range(len(files)):
#     file = files[i]
#     filepath = path + "/" + file
#     prefix = file.split('.')[0]
#     if os.path.isfile(filepath):
#         print('********    file   ********',file)
#         # img = cv2.imread('InputImages/' + file)
#         img = cv2.imread(folder + '/InputImages/' + file)
#         img = stretching(img)
#         sceneRadiance = sceneRadianceRGB(img)
#         # cv2.imwrite(folder + '/OutputImages/' + Number + 'Stretched.jpg', sceneRadiance)
#         sceneRadiance = HSVStretching(sceneRadiance)
#         sceneRadiance = sceneRadianceRGB(sceneRadiance)
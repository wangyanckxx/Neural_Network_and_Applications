from PIL import Image
import numpy as np
from scipy.io import wavfile
import scipy
import os, sys
from copy import deepcopy
import random 
import matplotlib.pyplot as plt
#from whitten import whitten
random.seed(0)
def rand(a,b):
    return (b - a) * random.random() + a #生成a到b的随机数


class ICA_data(object):
    def __init__(self):
        return 
    def Read_picture(self,Dict = '风景图\\'):
        #图像读取
        self.ab = []
        for i in range(1,5):
            Dictory = Dict+str(i)+'.bmp'
            temp = Image.open(Dictory)
            im = np.array(temp)
            self.imag_shape = im.shape
            ima = im.flatten()
            imag = ima
            if i == 1:
                self.imag_data = np.zeros(shape = (4,imag.shape[0]))
            self.imag_data[i-1,:] = imag
        return deepcopy(self.imag_data)


    def Mix_data(self,data,shap = (4,4)):

        Linear_trans = np.zeros(shap)
        for i in range(shap[0]):
            for j in range(shap[1]):
                temp = rand(1,5)
                Linear_trans[i,j] = temp
        Mix_data = np.dot(Linear_trans,data)
        return Mix_data

    def imag_produce(self,imag,Dict = 'imag_restoration\\'):
        #将分离后的信号还原成图片
        length = len(imag)
        for i in range(length):
            dictory = Dict +str(i)+'.bmp'
            os.makedirs(Dict,exist_ok=True)
            bmp_data = (np.array([imag[i]])).astype('uint8')
            new_im = Image.fromarray(bmp_data.reshape(self.imag_shape))
            new_im.save(dictory)
        return 

  

    def piceture_show(self,picture,title):      #将图片用matplot显示
        numb = len(picture)                     
                            #获取图像数目
        figure = plt.figure(1)      
                            #初始化
        for i in range(numb):
            img = picture[i].reshape(self.imag_shape)
                                        #图像还原
            plt.subplot(numb,1,i+1)
                                        #选择numb行，1列
            plt.imshow(img,cmap='gray')
                                        #显示灰度图像
            plt.xticks(())
            plt.yticks(())
           # plt.imshow(img)
        figure.suptitle(title)          #标题
        plt.show()                      #显示

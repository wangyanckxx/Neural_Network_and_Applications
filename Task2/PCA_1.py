#coding:utf8
from PIL import Image
import numpy as np
import math
from skimage.measure import compare_mse, compare_psnr,compare_ssim
import matplotlib
import matplotlib.pyplot as plt


def loadImage(path):
    img = Image.open(path)
    # 将图像转换成灰度图
    img = img.convert("L")
    # 图像的大小在size中是（宽，高）
    # 所以width取size的第一个值，height取第二个
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    # 为了避免溢出，这里对数据进行一个缩放，缩小100倍
    data = np.array(data).reshape(height,width)/100
    # data = np.array(data).reshape(height,width)
    # 查看原图的话，需要还原数据
    # print('type(data)',type(data))
    new_im = Image.fromarray(data*100)
    # new_im.show()
    return data



def pca(data,k):
    """
    :param data:  图像数据
    :param k: 保留前k个主成分
    """
    n_samples,n_features = data.shape
    # 求均值
    mean = np.array([np.mean(data[:,i]) for i in range(n_features)])
    # 去中心化
    normal_data = data - mean
    # 得到协方差矩阵
    matrix_ = np.dot(np.transpose(normal_data),normal_data)
    # 有时会出现复数特征值，导致无法继续计算，这里用了不同的图像，有时候会出现复数特征，但是经过
    # 我能知道的是协方差矩阵肯定是实对称矩阵的
    eig_val,eig_vec = np.linalg.eig(matrix_)
   # print(matrix_.shape)
   #  print(eig_val)
    # 第一种求前k个向量
   #  eig_pairs = [(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(n_features)]
   #  eig_pairs.sort(reverse=True)
   #  feature = np.array([ele[1] for ele in eig_pairs[:k]])
   #  new_data = np.dot(normal_data,np.transpose(feature))
    # 第二种求前k个向量
    eigIndex = np.argsort(eig_val)
    eigVecIndex = eigIndex[:-(k+1):-1]
    feature = eig_vec[:,eigVecIndex]
    # print('normal_data',normal_data)
    # print('normal_data.shape',normal_data.shape)
    new_data = np.dot(normal_data,feature)
    # print('feature',feature)
    # print('feature.shape',feature.shape)
    # print('new_data',new_data.shape)
    # 将降维后的数据映射回原空间
    rec_data = np.dot(new_data,np.transpose(feature))+ mean
    # print('rec_data',rec_data)
    # print('rec_data[0,0]',rec_data[0,0])
    # print('typerec_data[0,0]',type(rec_data[0,0]))
    # print('typerec_data[0,0]',np.abs(rec_data[0,0]))
    # # 压缩后的数据也需要乘100还原成RGB值的范围
    # print("type(rec_data)",type(rec_data))
    rec_data = np.abs(rec_data)
    # print('rec_data',rec_data)

    newImage = Image.fromarray(rec_data*100)
    # new_im = Image.fromarray(data*100)
    # PSNR = compare_mse(rec_data*100, data*100)
    # PSNR = 20 * math.log10(255 / math.sqrt(PSNR))
    # print('PSNR',PSNR)

    # newImage.show()
    return rec_data



def Error(data,recdata):
    sum1 = 0
    sum2 = 0
    D_value = data - recdata
    for i in range(data.shape[0]):
        sum1 += np.dot(data[i],data[i])
        sum2 += np.dot(D_value[i], D_value[i])
    error = sum2/sum1
    return error




data = loadImage("F:/Experiments/NeuralNetworkandApplication/Task2/koala.jpg")
Principal_Component_Number = []
PSNR_all = []
Error_all = []
for i in range(1,150,20):
    rec_data= pca(data,k=i)
    # PSNR = compare_mse(rec_data*100, data*100)
    # PSNR = 20 * math.log10(255 / math.sqrt(PSNR))

    # ssim = compare_ssim(rec_data*100, data*100, data_range=255)
    error = Error(data*100,rec_data*100)
    Error_all.append(error)

    Principal_Component_Number.append(i)
    # PSNR_all.append(ssim)

    # print('ssim',ssim)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Principal_Component_Number, Error_all, marker='^',color='y')
plt.xlabel('Principal Component Number')
plt.ylabel('Error')
plt.show()



# rec_data= pca(data,k=51)





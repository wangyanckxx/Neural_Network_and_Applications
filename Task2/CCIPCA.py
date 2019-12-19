from skimage.measure import compare_mse, compare_psnr, compare_ssim
import math
from PCA_data import PCA_data
from PIL import Image
import numpy as np
from scipy import linalg as la
from sklearn.utils import check_array, as_float_array
import matplotlib
import matplotlib.pyplot as plt


def loadImage(path):
    img = Image.open(path)
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    data = np.array(data).reshape(height, width)
    return data


def ccipca(X, k):
    n_samples, n_features = X.shape
    V = np.zeros((k, n_features))
    n = 1
    # 初始化均值向量为0均值的随机向量
    mt = np.random.randn(n_features)
    mt = mt - np.mean(mt)
    for j in range(len(X)):
        u = X[j]
        if n == 1:
            mt = mt + u
        else:
            mt = float(n - 1) * mt / n + u / float(n)
        u1 = u - mt
        for i in range(min(k, n)):
            if (i + 1) == n:
                V[i, :] = u1
            else:
                V[i, :] = (n - 1) * V[i, :] / float(n) + (np.linalg.norm(u1) ** 2) * V[i, :] / (
                n * np.linalg.norm(V[i, :]))
                u1 = u1 - np.sum(u1 * V[i, :].reshape(-1)) * V[i, :] / (np.linalg.norm(V[i, :]) ** 2)
        n = n + 1

    # W = V/np.linalg.norm(V, axis=1, keepdims=True)
    W = np.zeros((k, n_features))
    for q in range(len(V)):
        W[q, :] = V[q, :] / np.linalg.norm(V[q, :])

    return W, V, mt



def Error(data,recdata):
    sum1 = 0
    sum2 = 0
    D_value = data - recdata
    for i in range(data.shape[0]):
        sum1 += np.dot(data[i],data[i])
        sum2 += np.dot(D_value[i], D_value[i])
    error = sum2/sum1
    return error


def psnr(decompass_imag, imag_data):
    MSE = compare_mse(decompass_imag, imag_data)
    PSNR = 20 * math.log10(255 / math.sqrt(MSE))
    return PSNR


def Funssim(decompass_imag, imag_data):
    ssim = compare_ssim(decompass_imag, imag_data, data_range=255)
    return ssim

if __name__ == '__main__':

    path = 'F:/Experiments/NeuralNetworkandApplication/Task2/PCA-master/koala.jpg'
    img = Image.open(path)
    # 将图像转换成灰度图
    img = img.convert("L")
    # # 图像的大小在size中是（宽，高）
    # # 所以width取size的第一个值，height取第二个
    width = img.size[0]
    height = img.size[1]
    imag_data = img.getdata()
    imag_data = np.array(imag_data).reshape(height, width)
    # print('imag_data',imag_data)

    Principal_Component_Number = []
    PSNR_all = []
    ssim_all = []
    error_all = []
    for i in range(1, 300, 10):
        n_components = i
        W, V, mt = ccipca(imag_data, i)
        new_data = np.dot(imag_data, W.T)
        decompass_imag = np.dot(new_data, W)
        # Image.fromarray(decompass_imag).show()
        # PSNR =  psnr(decompass_imag, imag_data)

        # print("i ,PSNR",i,PSNR)


        # ssim = Funssim(decompass_imag, imag_data)
        error = Error(imag_data,decompass_imag)

        # newImage = Image.fromarray(decompass_imag)
        # newImage.show()


        Principal_Component_Number.append(i)
        # PSNR_all.append(PSNR)
        # ssim_all.append(ssim)
        error_all.append(error)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(Principal_Component_Number, PSNR_all,marker='^')
    # ax.plot(Principal_Component_Number, ssim_all,'r')
    ax.plot(Principal_Component_Number, error_all, 'y')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Error')
    plt.show()
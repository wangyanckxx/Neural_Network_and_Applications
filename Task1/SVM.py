import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# 使用管道将StandardScaler和SVC连在一起
from sklearn.pipeline import Pipeline
from sklearn import svm


import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV



def SVM_two(X,y):

    colum = X.shape
    x = np.array([X]).reshape(colum[0],1)


    svc1 = SVR(kernel='rbf', C=1, gamma=0.1)
    y_svc1 = svc1.fit(x, y).predict(x)

    svc2 = SVR(kernel='rbf', C=1, gamma=1)
    y_svc2 = svc2.fit(x, y).predict(x)

    svc3 = SVR(kernel='rbf', C=1, gamma=10)
    y_svc3 = svc3.fit(x, y).predict(x)

    svc4 = SVR(kernel='rbf', C=1, gamma=100)
    y_svc4 = svc4.fit(x, y).predict(x)

    svc5 = SVR(kernel='rbf', C=5, gamma=0.1)
    y_svc5 = svc5.fit(x, y).predict(x)


    svc6 = SVR(kernel='rbf', C=5, gamma=1)
    y_svc6 = svc6.fit(x, y).predict(x)


    svc7 = SVR(kernel='rbf', C=5, gamma=10)
    y_svc7 = svc7.fit(x, y).predict(x)


    svc8 = SVR(kernel='rbf', C=5, gamma=100)
    y_svc8 = svc8.fit(x, y).predict(x)


    flg = plt.figure()
    # flg.subplots_adjust(left=0.15,bottom=0.1,top=0.9,right=0.95,hspace=0.35,wspace=0.25)
    plt.scatter(x, y, color='darkorange', label='data')
    plt.subplot(2, 4, 1), plt.plot(x, y, 'b*',x, y_svc1, 'r.', linewidth=3.0),plt.title('gamma=0.1,C=1')

    plt.subplot(2, 4, 2), plt.plot(x, y, 'b*',x, y_svc2, 'r.', linewidth=3.0),plt.title('gamma=1,C=1')

    plt.subplot(2, 4, 3), plt.plot(x, y,'b*', x, y_svc3, 'r.', linewidth=3.0),plt.title('gamma=10,C=1')
    plt.subplot(2, 4, 4),plt.plot(x, y, 'b*',x, y_svc4, 'r.', linewidth=3.0), plt.title('gamma=100,C=1')
    plt.subplot(2, 4, 5), plt.plot(x, y,'b*', x, y_svc5, 'r.', linewidth=3.0),plt.title('gamma=0.1,C=5')

    plt.subplot(2, 4, 6), plt.plot(x, y,'b*', x, y_svc6, 'r.', linewidth=3.0),plt.title('gamma=1,C=5')

    plt.subplot(2, 4, 7),plt.plot(x, y,'b*', x, y_svc7, 'r.', linewidth=3.0), plt.title('gamma=10,C=5')

    plt.subplot(2, 4, 8), plt.plot(x, y,'b*', x, y_svc8, 'r.', linewidth=3.0),plt.title('gamma=100,C=5')

    plt.show()






#
#
# svr_rbf = SVR(kernel='rbf', C=1, gamma=100)
#
#
# svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=5)
# # svr_GridSearch = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
# y_rbf = svr_rbf.fit(X, y).predict(X)
# y_lin = svr_lin.fit(X, y).predict(X)
# y_poly = svr_poly.fit(X, y).predict(X)
# # y_GridSearch =svr_GridSearch.fit(X, y).predict(X)
#
# ###############################################################################
# # look at the results
# lw = 2
# plt.scatter(X, y, color='darkorange', label='data')
# plt.hold('on')
# plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
# plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
# plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
# # plt.plot(X, y_GridSearch, color='r', lw=lw, label='svr_GridSearch')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()
#
# from sklearn import svm
# X = [[0], [2]]
# y = [0.5, 2.5]
# clf = svm.SVR()
# clf.fit(X, y)
#
# print("clf.predict([[1, 1]])",clf.predict([[1, 1]]))



#
#
# x = [[0,0], [2,2]]
#
# y = [0.5, 2.5]
#
# clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto_deprecated',
# kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# clf.fit(x, y)
#
#
# plt.figure()
# plt.plot(x, y, color='blue', linewidth=3.0, label="test")
# plt.plot(x, clf.predict(x), color='red', linestyle='--', linewidth=3.0)
# plt.legend(["Orignal pointers", "RBF"], loc='best')
# plt.show()




#
# clf.predict([[1, 1]])
#
#
#
#
# x, y = datasets.make_moons(noise=0.15, random_state=100)
# print(x)
# print('x[y == 0, 0], x[y == 0, 1]',x[y == 0, 0], x[y == 0, 1])
# print('x[y == 1, 0], x[y == 1, 1]',x[y == 1, 0], x[y == 1, 1])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.show()
#
#
# def RBFKernelSVC(gamma, C):
#     return Pipeline([
#         ('std_scaler', StandardScaler()),
#         # 采用高斯核函数rbf
#         # gamma越大，高斯图形越窄，模型复杂度越高，容易导致过拟合
#         # gamma越小，高斯图形越宽，模型复杂度越低，容易导致欠拟合
#         ('svc', SVC(kernel='rbf', gamma=gamma, C=C))
#     ])
#
#
# svc1 = RBFKernelSVC(0.1, 1)
# svc1.fit(x, y)
# svc2 = RBFKernelSVC(1, 1)
# svc2.fit(x, y)
# svc3 = RBFKernelSVC(10, 1)
# svc3.fit(x, y)
# svc4 = RBFKernelSVC(100, 1)
# svc4.fit(x, y)
#
# svc5 = RBFKernelSVC(0.1, 5)
# svc5.fit(x, y)
# svc6 = RBFKernelSVC(1, 5)
# svc6.fit(x, y)
# svc7 = RBFKernelSVC(10, 5)
# svc7.fit(x, y)
# svc8 = RBFKernelSVC(100, 5)
# svc8.fit(x, y)
#
#
# def plot_decision_boundary(model, axis):
#     x0, x1 = np.meshgrid(
#         np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)),
#         np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100))
#     )
#     x_new = np.c_[x0.ravel(), x1.ravel()]
#     y_predict = model.predict(x_new).reshape(x0.shape)
#
#     from matplotlib.colors import ListedColormap
#     # 自定义colormap
#     custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
#     plt.contourf(x0, x1, y_predict, linewidth=5, cmap=custom_cmap)
#
#
# flg = plt.figure()
# # flg.subplots_adjust(left=0.15,bottom=0.1,top=0.9,right=0.95,hspace=0.35,wspace=0.25)
# plt.subplot(2, 4, 1), plt.title('gamma=0.1,C=1')
# plot_decision_boundary(svc1, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.subplot(2, 4, 2), plt.title('gamma=1,C=1')
# plot_decision_boundary(svc2, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.subplot(2, 4, 3), plt.title('gamma=10,C=1')
# plot_decision_boundary(svc3, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.subplot(2, 4, 4), plt.title('gamma=100,C=1')
# plot_decision_boundary(svc4, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.subplot(2, 4, 5), plt.title('gamma=0.1,C=5')
# plot_decision_boundary(svc5, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.subplot(2, 4, 6), plt.title('gamma=1,C=5')
# plot_decision_boundary(svc6, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.subplot(2, 4, 7), plt.title('gamma=10,C=5')
# plot_decision_boundary(svc7, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.subplot(2, 4, 8), plt.title('gamma=100,C=5')
# plot_decision_boundary(svc8, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(x[y == 0, 0], x[y == 0, 1])
# plt.scatter(x[y == 1, 0], x[y == 1, 1])
# plt.show()
#
#
#
#

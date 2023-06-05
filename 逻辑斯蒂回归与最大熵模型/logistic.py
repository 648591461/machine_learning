import numpy as np
import pandas as pd

"""Sigmod"""
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))  # P(Y = 0|X)

"""梯度上升算法"""
"""
函数说明:梯度上升算法
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                        #最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error  # 梯度更新公式 w_(t + 1) = w_t + α * (w_t * x_i - y_i)^T * x_i
    return weights.getA()                                                #将矩阵转换为数组，返回权重数组

if __name__ == '__main__':
    # 预测模型：逻辑斯蒂回归模型
    # 优化策略：极大似然估计
    # 优化算法：梯度下降算法
    # 二分类问题实现

    # 数据读取：采用kaggle的心脏疾病数据集
    all_df = pd.read_csv('heart.csv')
    feature = np.array(all_df.drop('target', 1))
    label = np.array(all_df['target'])

    # 权重更新
    weight = gradAscent(feature, label)
    # 预测
    pre = sigmoid(np.matmul(feature, weight)).astype(int)
    x = pre == np.expand_dims(label, axis=-1)
    print(sum(x) / len(label))
# TODO 2 手撕最大熵模型实现分类 牛顿法 拟牛顿法




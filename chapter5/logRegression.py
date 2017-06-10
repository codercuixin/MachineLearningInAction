# *--coding:utf-8--*
from numpy import *;

'''
实现Logistics回归
'''
def loadDataSet():
    dataMat = [] ; labelMat = [];
    for line in open("testSet.txt").readlines():
        lineArr = line.strip().split();
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmod(inX):
    '''
    近似的阶跃函数，当x值为0时为0.5,x值越大越接近1，x值越小越接近0
    :param inX:
    :return:
    '''
    return 1/(1+exp(-inX))

def gradAscent(dataMatIn, classLables):

    '''
    梯度上升算法
    :param dataMatIn:
    :param classLables:
    :return:
    '''
    dataMatrix =mat(dataMatIn)
    labelMat = mat(classLables).transpose();
    m,n= shape(dataMatrix)
    alpha = 0.001; #步长
    maxCycle = 500
    weights= ones((n,1))

    for k in range(maxCycle):
        h = sigmod(dataMatrix * weights) #估计的结果
        error = labelMat - h #计算出误差
        weights = weights + alpha* dataMatrix.transpose() * error
    return  weights
def stogGradAscent0(dataMatrix, classLabels):
    '''
    随机梯度算法
    所有回归系数初始化为1
    对数据集中每个样本
        计算该样本的梯度
        使用alpha*gradient更新回归系数值
    返回回归系数值
    :param dataMatrix:
    :param classLabels:
    :return:
    '''
    m, n =  shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m) :
        h = sigmod(sum(dataMatrix[i] *  weights))
        error = classLabels[i] - h
        weights = weights + alpha *error* dataMatrix[i]
    return weights
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for i in range(numIter):
        dataIndexes = range(m)
        for j in range(m):
            #update alpha
            alpha = 4/(1.0+i+j) + 0.01;

            randomIndex = int(random.uniform(0, len(dataIndexes)))
            h= sigmod(sum(dataMatrix[randomIndex] * weights))
            error = classLabels[randomIndex] - h
            weights = weights + alpha * error * dataMatrix[randomIndex]
            del(dataIndexes[randomIndex])
    return weights
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [] ;ycord1 = [];
    xcord2 = []; ycord2 = [];
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    x = arange(-3.0, 3.0, 0.1) #x轴代表X1
    y = (-weights[0] -weights[1] *x)/weights[2] #y轴代表X2
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifyVector(inX, weights):
    '''
    将每个特征向量乘以回归系数，再将该结果求和，
    如果大于0.5,返回1.0；否则，返回0.0
    :param inX:
    :param weights:
    :return:
    '''
    prob = sigmod(sum(inX * weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def cocliTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainMat=[]; trainLabels=[]
    for line in frTrain.readlines():
        tempArr = line.strip().split('\t')
        lineArr = []
        for i in range(21): #前20个特征
            lineArr.append(float(tempArr[i]))
        trainMat.append(lineArr)
        trainLabels.append(float(tempArr[21]))
    trainWeights = stocGradAscent1(array(trainMat), trainLabels, 500)

    error = 0.0 ;
    count = 0.0;
    for line in frTest.readlines():
        tempArr = line.strip().split('\t')
        count += 1
        lineArr = []
        for i in range(21): #前20个特征
            lineArr.append(float(tempArr[i]))
        classifyResult = classifyVector(lineArr, trainWeights)
        if int(classifyResult) != int(tempArr[21]):
            error += 1
    print "the error rate is %f:" %(error/count)
    return  error/count

def mutiTest():

    numTests = 10; avgErrorRate = 0.0
    for i in range(numTests):
        avgErrorRate += cocliTest()
    avgErrorRate = avgErrorRate/numTests
    print "after  %d iterations the average rate is %f" %(numTests, avgErrorRate)



if __name__ == "__main__":
    # dataMat, labelMat = loadDataSet()
    # weights= gradAscent(dataMat, labelMat)
    # print weights
    # plotBestFit(weights.getA())

    # dataArr , labelMat = loadDataSet()
    # weights = stogGradAscent0(array(dataArr), labelMat)
    # plotBestFit(weights)

    # dataArr , labelMat = loadDataSet()
    # weights = stocGradAscent1(array(dataArr), labelMat)
    # plotBestFit(weights)

    mutiTest()



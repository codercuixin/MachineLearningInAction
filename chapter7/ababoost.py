#!--**coding:utf-8**--
from  numpy import *;
def loadSimpleData():
    datMat = matrix([
        [1., 2.1],
        [2, 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值进行分类，
    :param dataMatrix:
    :param dimen: 要分类的某一列
    :param threshVal: 阈值
    :param threshIneq: lt表示小于等于的要
    :return:
    '''
    retArray = ones((shape(dataMatrix)[0], 1)) #首先将retArray全置为1
    #接着过滤出符合要求的数组下标，进而将其置为-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1;
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1;
    return retArray
def buildStump(dataMatrix, classLabels, D):
    '''
    将最小错误率minError置为无穷大
    对数据集中的每个特征（每一列）第一层循环：
        对每个步长，第二层循环：
            对每个不等号，第三层循环：
                建立一颗单层决策树，并利用加权数据集对它测试
                如果错误率低于minError，则将当前单层决策树稍微最佳单层决策树
    返回最佳单层决策树。
    :param dataMatrix:
    :param classLabels:
    :param D:权重列向量
    :return:返回具有最小错误率的单层决策树，错误率，估计的类别向量。
    '''
    dataMatrix = mat(dataMatrix);
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10; bestStump = {}; bestClassEst = mat(zeros((m,1)))
    minError = inf #无穷大
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()#第i列的最小值
        rangeMax = dataMatrix[:, i].max()#第i列最大值
        stepSize = (rangeMax - rangeMin)/numSteps

        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize) #阈值
                predicitedVals = stumpClassify(dataMatrix, i, threshVal, inequal)

                errorArr = mat(ones((m, 1)))
                errorArr[predicitedVals == labelMat] = 0; #将errorArr中，预测与实际相等的还原为0
                #计算加权错误率
                weightedError = D.T * errorArr #将权重列向量的转置与errorArr矩阵内积，得到weightedError标值

                # print "split: dim %d, thresh %.2f, thresh inequal %s, the weighted error is %.3f" \
                #       %(i, threshVal, inequal, weightedError)
                if(weightedError < minError):
                    minError = weightedError;
                    bestClassEst = predicitedVals.copy();
                    bestStump['dim'] = i #保存当前最佳单层决策树
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDs(dataArr, classLabels, numIt=40):
    '''

    :param dataArr: 数据集
    :param classLabels: 类别标签dbarray
    :param numIt: 迭代次数
    :return:
    '''
    weakClassArr = [] ; #用来存储每一次的bestStump(最好树桩
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)#初始化权重列向量，每个元素变为1/m
    aggClassEst = mat(zeros((m,1))) #记录每个数据点估计累计值
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:",D.T
        alpha = float(0.5 * log((1.0-error)/max(error, 1e-16)))
        #利用书上的公式a = 1/2ln((1-error) / error)来计算alpha，其中max(error, 1e-16)防止发生除零溢出

        bestStump['alpha'] = alpha;
        weakClassArr.append(bestStump) #将bestStump字典放到列表中
        print "ClassEst:",classEst.T

        #利用公式Di+1 = Di* e的(+-alpha)次方/sum(D)来下一次迭代的权重列向量D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  #计算出expon列向量，
        # 如果真实结果与实际结果相同（因为只有正负一）， 为-alpha;如果真实结果与实际结果不同，为alpha,
        D = multiply(D, exp(expon)) #两个列向量对应元素相乘,通过exp使得权重发生变化，错误的权重变大，正确的权重变小
        D = D/D.sum() #D.sum会将矩阵所有元素相加

        #错误率累加计算
        aggClassEst += alpha * classEst
        print "aggClassEst:", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        #通过将累加分类器结果的aggClassEst利用sign取符号，与classLabels转换成的列向量比较的每个元素比较，如果不等于则为TRUE，等于则为False
        errorRate = aggErrors.sum()/m
        print "total error :", errorRate, "\n"

        #如果错误率为零，就跳出
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst
def adaClassify(dataToClassify, classifierArr):
    dataMatrix = mat(dataToClassify)
    m = shape(dataMatrix)[0];
    aggClassEst = mat(zeros((m, 1))) #用来统计累加所有的类别估计值

    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)
def loadDataSet(fileName):

    featureNum = len(open(fileName).readline().split('\t'))
    dataMat =[] ; labelMat = []
    file = open(fileName)
    for line in file.readlines():
        lineArr = line.split('\t');
        dataRow = []
        for i in range(featureNum -1):
            dataRow.append(float(lineArr[i]))
        dataMat.append(dataRow)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0) #在（0,0）和（1,1）与坐标轴围城的正方形内作图

    ySum = 0.0 #用来累加每个正方形的y值，为最后计算AUC做准备

    numPosClass = sum(array(classLabels) == 1.0) #通过数组过滤的方法得到正列的个数,sum放在外面只是用来数True的个数

    yStep = 1/(float(numPosClass)) #计算X轴，Y轴的步长
    xStep = 1/(float(len(classLabels) - numPosClass))

    sortedIndicates = argsort(predStrengths)#返回排序索引，因为是从小到大的，因此从<1,1>画到<0,0>

    #构建画笔
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sortedIndicates.tolist()[0]:
        if (classLabels[index] == 1.0):
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0;
            ySum += cur[1] #累计求矩形的高度
        ax.plot([cur[0],cur[0]-delX], [cur[1], cur[1]-delY], c='g') #将新的点与原来的点连接起来，先是x轴坐标，后是y轴坐标,最后是颜色
        cur = (cur[0]-delX, cur[1]-delY)#更新当前节点为新的点
    ax.plot([0,1], [0,1], 'b--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print 'the Area under the curve is: ', ySum * xStep; #打印出AUC







if __name__ == "__main__":
    # D = mat(ones((5, 1))/5)
    # dataMat, classLabels = loadSimpleData();
    #
    # # buildStump(dataMat, classLabels, D) #测试buildStump
    #
    # weakClassArr = adaBoostTrainDs(dataMat, classLabels, 30)
    # print weakClassArr
    #
    # #test adaClassify
    # print adaClassify([[0, 0], [5, 5]], weakClassArr)


    #使用Adaptive Boosting来预测马匹
    # dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    # print dataArr
    # print labelArr
    # classifierDic = adaBoostTrainDs(dataArr, labelArr,10)
    # print classifierDic
    # testDataArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # predictionVals = adaClassify(testDataArr, classifierDic)
    # errorVals = multiply(predictionVals != mat(testLabelArr).T, ones((67,1)))
    # print errorVals.sum(), errorVals.sum()/67

    #测试ROC工作情况
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    print dataArr
    print labelArr
    classifierDic, aggClassEst= adaBoostTrainDs(dataArr, labelArr,10)
    plotROC(aggClassEst.T, labelArr)
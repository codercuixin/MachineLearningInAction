#!--**coding:utf-8**--
import random;
from numpy import  *;
def loadDataSet(fileName):
    '''
    get the data set
    :param fileName:
    :return:
    '''
    dataMat = []; lableMat = [];
    fr  = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t');
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        lableMat.append(float(lineArr[2]))
    return dataMat, lableMat
def selectJrand(i, m):
    '''
    select a random number between 0 and m
    :param i:
    :param m:
    :return:
    '''
    j = i;
    while(j == i):
        j = int(random.uniform(0, m))
    return j;

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H;
    if L > aj:
        aj = L;
    return aj;
def smoSimple(dataMatIn, classLable, C, toler, maxIter):
    '''
    创建一个alpha向量，并将其初始化为0向量
    当迭代次数少于最大迭代次数时(外循环)
        对数据及中的每个数据向量(内循环)
            如果该数据向量可以被优化：
                随机选择另外一个数据向量
                同时优化这两个向量
                如果两个向量都不能优化，退出内循环
        如果所有向量都没被优化，增加迭代数目，继续下一次循环
    :param dataMatIn: 数据集
    :param classLable: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前的最大循环次数
    :return:
    '''
    dataMatrix = mat(dataMatIn);
    labelMat = mat(classLable).transpose();
    b= 0;
    m, n =  shape(dataMatrix) #m,n分别代表数据集的行和列
    alphas = mat(zeros((m,1)))
    iter = 0;
    while(iter < maxIter):
        alphaPairChanged = 0;
        for i in range(m):
            # mutiply([m, 1], [m, 1]).T = [m,1].T = [1, m]
            # [m, n] *[1, n].T = [m, n] * [n, 1] = [m, 1]
            #[1, m] * [m, 1] = [1, 1]
             #fXi表示预测的类别
            fXi = float(multiply(alphas,labelMat).T*\
                        (dataMatrix * dataMatrix[i,:].T)) + b;
            #误差，预测结果-真实结果
            Ei = fXi - float(labelMat[i])

            #如果可以优化的话
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i] *Ei > toler ) and (alphas[i] > 0)):
                #随机选择第二个alpha
                j = selectJrand(i, m);
                fXj = float(multiply(alphas, labelMat).T*\
                            (dataMatrix * dataMatrix[j, :].T)) + b
                #计算其误差
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy()
                #保证alpha在0，C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+ alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H : print "L == H "; continue;

                #eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T\
                        - dataMatrix[i, :] * dataMatrix[i,:].T\
                        - dataMatrix[j, :] * dataMatrix[j, :].T;
                if eta >= 0: print "ETA>=0"; continue;

                #对alphas[j]进行调整
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta;
                alphas[j] = clipAlpha(alphas[j], H, L)
                if( abs(alphas[j] - alphaJold) <0.00001 ): print \
                    "j not moving enough"; continue;

                 #对alphas[i]进行调整
                alphas[i] += labelMat[j] * labelMat[i]*(\
                    alphaJold - alphas[j])

                #设置常数项b
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)\
                     *dataMatrix[i,:]*dataMatrix[i,:].T -\
                    labelMat[j]*(alphas[j] - alphaJold)*\
                     dataMatrix[i,:] *dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)\
                     *dataMatrix[i,:]*dataMatrix[j,:].T -\
                    labelMat[j]*(alphas[j] - alphaJold)*\
                     dataMatrix[j,:] *dataMatrix[j,:].T

                if(0 < alphas[i]) and (C > alphas[i]) : b = b1;
                elif (0 < alphas[j] ) and (C > alphas[j]) : b = b2;
                else : b = (b1 + b2)/2


                alphaPairChanged += 1
                print "iter: %d i:%d, pairs changed %d" %\
                      (iter, i, alphaPairChanged)
        if(alphaPairChanged == 0 ): iter += 1
        else: iter = 0;
        print "iteration number: %d" %(iter)
    return b, alphas


class optStruct:
    '''
    定义一个类来保存所有用要的值，
    '''
    def __init__(self, dataMatIn, classLabel, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabel
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0;
        self.eCache = mat(zeros((self.m, 2))) #误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup);
def calcEk(oS, k):
    '''
    对于给定的alpha值，就算E值并返回
    :param oS:
    :param k:
    :return:
    '''
    fXk = float(multiply(oS.alphas, oS.labelMat).T *\
          oS.K[:, k] +oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    '''
    用于选择第二个alpha或者说是内循环的alpha值
    :param i:
    :param oS:
    :param Ei:
    :return:
    '''
    maxK = -1;  maxDeltaE = 0; Ej = 0;
    oS.eCache[i] = [i, Ei] #将Ei所在行设置为有效的
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0] #将ECache中第一列中非零元素的下标合并一个列表，然后返回
    if(len(validEcacheList) > 1): #如果不是第一次SelectJ的话
        for k in validEcacheList:
            if k == i:continue
            Ek = calcEk(oS, k);
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek;
        return maxK, Ej; #返回步长最大的j
    else:#如果是第一次进入的haul
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
def updateEk(oS, k):
    '''
    更新k行的ECache值
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEk(oS, k);
    oS.eCache[k] = [1, Ek]
def innerL(i, oS):
    Ei = calcEk(oS, i);
    if((oS.labelMat[i]*Ei < - oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > oS.C)):
        j, Ej = selectJ(i, oS, Ei);#选择最大步长的j
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if( L == H): print "L==H";return 0;
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if(eta >=0): print "eta>=0"; return 0

        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)

        updateEk(oS, j) #更新误差缓存

        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough";return 0;

        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] *(alphaJold - alphaIold)
        updateEk(oS, i) #更新i的误差缓存

        b1 = oS.b - Ei - \
             oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - \
             oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1;
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2;
        else: oS.b = (b1 + b2)/2.0
        return 1;
    else:
        return 0;
def smoP(dataMatIn, classLables, C, toler, maxIter, KTup=("lin", 0)):
    '''

    :param dataMatIn:
    :param classLables:
    :param C:
    :param toler:
    :param maxIter:
    :param KTup:  数组第一个为采用的核函数的名称lin 还是rbf,第二个参数只有rbf时才会用到，表示函数值跌落到0的速率
    :return:
    '''
    oS = optStruct(mat(dataMatIn), mat(classLables).transpose(), C, toler, KTup)
    iter =0;
    entiresSet = True;
    alphaPairChanged = 0;
    while( iter < maxIter and alphaPairChanged > 0) or (entiresSet):
        alphaPairChanged = 0;
        if entiresSet:
            #遍历所有的值
            for i in range(oS.m):
                alphaPairChanged += innerL(i, oS)
                print "fullSet , iter: %d, i: %d,  paris changed %d" %(iter, i, alphaPairChanged)
            iter += 1;
        else:
            #遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A>0) * oS.alphas.A<C)[0]
            for i in nonBoundIs:
                alphaPairChanged += innerL(i, oS)
                print "non-bound, iter: %d, i: %d, paris changed %d" %(iter, i, alphaPairChanged)
            iter += 1

        if entiresSet: entiresSet = False;
        elif alphaPairChanged ==0 : entiresSet = True;

        print "iteration number ; %d" %(iter)
    return oS.b, oS.alphas
def calcWs(alphas, dataArr, classLabels):
    '''
    计算权重列向量
    :param alphas:
    :param dataArr:
    :param classLabels:
    :return:
    '''
    X = mat(dataArr);
    labelMat = mat(classLabels).transpose();
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T);
    return w;

def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':  K = X * A.T #如果是线性核函数的话
    elif kTup[0] == 'rbf': #如果径向基核函数
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T #内积，这两个向量相乘得到标值或者数值
        K= exp( K / -1 * kTup[1] **2) #计算高斯版本的径向基函数的数值
    else: raise "Opps the kernel is not supproted for now"
    return K
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
def loadImages(dirName):
    from os import  listdir
    hwLables = [];
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024)) #shape是(m, 1024)的
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0];
        classNameStr = int(fileStr.split('_')[0])
        if classNameStr == 9: hwLables.append(-1) #只区分是9与不是9两种情况
        else: hwLables.append(1)
        trainingMat[i,:] = img2vector(dirName+"/"+fileNameStr)
    return trainingMat, hwLables

def testline():
    dataArr, labelArr = loadDataSet("testSet.txt")
    print labelArr;
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # print b
    # print alphas[alphas>0] # 返回大于0的元素
    # print shape(alphas[alphas>0]) #返回支持向量的个数
    #
    # for i in range(100): #打印支持向量
    #     if(alphas[i] > 0):
    #         print dataArr[i], labelArr[i];

    #测试权重列向量
    ws = calcWs(alphas, dataArr, labelArr)
    print ws

    #测试分类
    dataMat = mat(dataArr)
    print dataMat[0] * mat(ws) + b #大于0表示属于1类，小于表示-1类
    print labelArr[0]

def testrbf(k1 = 1.3):
    '''
    测试径向基核函数
    :param k1: 表示函数值跌落到0的速度参数
    :return:
    '''
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))

    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose() #行向量转为列向量
    svInd = nonzero(alphas.A> 0) [0]
    sVs = dataMat[svInd]
    labelSv = labelMat[svInd]
    print "there are %d support Vectors" % shape(sVs)[0]
    m, n = shape(dataMat)

    errorCount = 0;
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is %f" %  (float(errorCount)/m)

    dataArr, labelArr = loadDataSet("testSetRBF2.txt")
    errorCount = 0
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose();
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is %f" %  (float(errorCount)/m)
def testDigits(k1 = 10):
    '''
    测试径向基核函数
    :param k1: 表示函数值跌落到0的速度参数
    :return:
    '''
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))

    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose() #行向量转为列向量
    svInd = nonzero(alphas.A> 0) [0]
    sVs = dataMat[svInd]
    labelSv = labelMat[svInd]
    print "there are %d support Vectors" % shape(sVs)[0]
    m, n = shape(dataMat)

    errorCount = 0;
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is %f" %  (float(errorCount)/m)

    dataArr, labelArr = loadImages("testDigits")
    errorCount = 0
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose();
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is %f" %  (float(errorCount)/m)

if __name__ == "__main__":
   #  # testrbf()
   testDigits()


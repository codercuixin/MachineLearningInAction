#!--**coding:utf8**--
from numpy import *

def loadDataSet(fileName):
    numberOfFeature = len(open(fileName).readline().split('\t')) -1
    dataArr = [];labelArr = []
    for line in open(fileName).readlines():
        lineArr = line.split('\t')
        lineData = []
        for i in range(numberOfFeature):
            lineData.append(float(lineArr[i]))
        dataArr.append(lineData)
        labelArr.append(float(lineArr[-1]))
    return dataArr, labelArr

def standRegression(xArr, yArr):
    '''
    通过w = (X.T*X).I *X.T * Y求出最优解
    使用库函数linalg.det来判断是否可逆
    linalg.det(xTx) == 0: #如果xTx的行列式为零的话，不可以求逆
    :param xArr:
    :param yArr:
    :return:
    '''
    xMat = mat(xArr);
    yMat = mat(yArr).T;
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0: #如果xTx的行列式为零的话，不可以求逆
        print  "这个矩阵是不可逆的"
        return
    wBest = xTx.I * (xMat.T * yMat) #利用公式求出可以估计的w的最优解
    return wBest
def testStandRegression():
    xArr, yArr = loadDataSet('ex0.txt')
    # print xArr,yArr
    wBest = standRegression(xArr, yArr)
    # print wBest
    #使用获得的最好系数，来重新算一下y值
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yPredict = xMat * wBest
    # print yPredict
    #通过计算预测值和真实值之间的相关系数，来评价这个模型对不同数据的分类效果。
    #相关系数，通过命令corrcoef(yEstimate, yActual) #要保证两个都是行向量
    print corrcoef(yPredict.T, yMat.T) #Correlation coefficient 相关系数
    #[[ 1.          0.98647356] [ 0.98647356  1.        ]] 

    #在自己的训练数据上好很正常，下面测试在另一组数据的相关性
    xArr2, yArr2 = loadDataSet('ex1.txt')
    yPredict2 = mat(xArr2) * wBest
    print corrcoef(mat(yArr2), yPredict2.T) #保证两个都是行向量，
    #[[ 1.          0.98524753][ 0.98524753  1.        ]]虽然下降了一点，但也还可以


    #以下是绘制原来的数据和最佳拟合直线的过程
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # #绘制原始数据
    # ax.scatter(xMat[:,1].flatten().A[0], yMat[:, 0].flatten().A[0])
    # # <matplotlib.collections.CicrleCollection object at 0x04ED9D30>
    #
    # xCopy = xMat.copy()
    # xCopy.sort(0) #因为需要绘制最佳拟合曲线，并且如果数据混轮的话，绘图会出现问题
    # yHat = xCopy * wBest
    # ax.plot(xCopy[:, 1], yHat)
    # plt.show()

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m))) #创建一个对角矩阵

    for i in range(m):
        diffMat = testPoint - xMat[i, :];
        weights[i, i] = exp((diffMat * diffMat.T)/ (-2.0 * k ** 2))
    xWx = xMat.T * weights * xMat;
    if linalg.det(xWx) == 0 : #如果横列式等于0的话，那么就不能求逆
        print "the matrix cannot inverse because of it is singular"
        return
    wBest = xWx.I * xMat.T * weights * yMat;
    return testPoint * wBest;

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0];
    yHat = zeros(m) #预计分类值
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat;
def testLwlr():
    xArr, yArr = loadDataSet('ex0.txt')
    # yHat = lwlrTest(xArr, xArr, yArr, 1.0)
    # yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    print yHat
    xMat = mat(xArr)
    srtId = xMat[:, 1].argsort(0)
    xSort = xMat[srtId][:, 0, :]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtId])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()
def testLwlr2():
    #测试海蛎数据
    abX ,abY = loadDataSet('abalone.txt')
    #在老数据上的表现
    yHat0 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 0.1)
    yHat1 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 1)
    yHat2 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 10)

    print resError(abY[0: 99], yHat0.T)
    print resError(abY[0: 99], yHat1.T)
    print resError(abY[0: 99], yHat2.T)
    #在新数据上的表现
    yHat0 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 0.1)
    yHat1 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 1)
    yHat2 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 10)

    print resError(abY[100: 199], yHat0.T)
    print resError(abY[100: 199], yHat1.T)
    print resError(abY[100: 199], yHat2.T)

    #与以前的线性回归作比较
    weightBest = standRegression(abX[0: 99], abY[0:99])
    yHat = mat(abX[100: 199]) * weightBest
    print resError(abY[100: 199], yHat.T.A)
def resError(yArr, yHatArr):
    '''
    计算误差的大小，均方差求和
    :param yArr: 真实分类数组
    :param yHatArr: 预测分类数组
    :return:
    '''
    return ((yArr - yHatArr)**2).sum()

def ridgeRegression(xMat, yMat, lam=0.2):
    '''
    岭回归
    :param xMat:
    :param yMat:
    :param lam:通过引入该lamd减少不重要的参数，这种技术在统计学中叫做缩减
    :return:
    '''
    xTx = xMat.T * xMat;
    denom = xTx + lam * eye(shape(xMat)[1]) ; #加上一个n*n 的单位矩阵
    if linalg.det(denom) == 0 :
        print "the matrix is singular, so it cannot  inverse"
        return
    weightBest = denom.I * xMat.T * yMat
    return weightBest
def ridgeTest(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    #标准化xMat,
    xMean = mean(xMat, 0) #axis= 0 算的是每一列的均值，得到是一个长度为m的array
    xVar = var(xMat, 0)
    xMat = (xMat - xMean)/xVar

    #标准化yMat
    yMat = yMat - mean(yMat, 0)

    numTestPst = 30;
    weightMat = zeros((numTestPst, shape(xMat)[1]))
    for i in range(numTestPst):
        weightBest = ridgeRegression(xMat, yMat, exp(i-10))
        weightMat[i,:] = weightBest.T
    return weightMat;
def testRidge():
    abX, abY = loadDataSet('abalone.txt')
    weightsMat = ridgeTest(abX, abY)
    print weightsMat
    #画图显示效果
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(weightsMat)
    plt.show()
def regularize(xMat):
    '''
    正规化xMat,减去均值除以方差
    :param xMat:
    :return:
    '''
    xCopy = xMat.copy()
    xMean = mean(xCopy, 0);
    xVar = var(xCopy, 0)
    xCopy = (xCopy - xMean)/xVar
    return xCopy
def stageWise(xArr, yArr, eps=0.01, numIt = 100):
    '''
    前向逐步线性回归
    :param xArr: 输入数据
    :param yArr: 标签数组
    :param eps: 每次迭代需要调整的步长
    :param numIt: 循环次数
    :return:
    '''
    #标准化xMat,yMat
    xMat = mat(xArr); yMat = mat(yArr).T
    xMat = regularize(xMat)
    yMat -= mean(yMat, 0)

    m, n = shape(xMat)
    returnMat = zeros((numIt, n)) #用来保存每一轮求得的最好回归参数

    ws = zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print ws.T #方便观察结果的变化
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                resE = resError(yMat.A, yTest.A)
                if resE < lowestError:
                    lowestError = resE;
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T #将这一轮求的最好回归参数保存下来
    return returnMat
def testStageWise():
    xArr, yArr = loadDataSet('abalone.txt')
    # stageWise(xArr, yArr, 0.01, 200)
    #采用更小的步长，更多的迭代次数
    returnMat = stageWise(xArr, yArr, 0.001, 5000)
    #与最小二乘法进行比较
    xMat = mat(xArr); yMat = mat(yArr).T
    xMat = regularize(xMat)
    yMat = yMat - mean(yMat, 0)
    weights = standRegression(xMat, yMat.T)
    print weights.T

    import  matplotlib.pyplot as plt;
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot(returnMat)
    plt.show()

def searchForSet(retX, retY, setNum, yr, numPce ,origPrc):
    from time import  sleep
    import  json
    import urllib2
    sleep(10)
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        trainX = []; trainY = [];
        testX = []; testY= [];
        random.shuffle(indexList)

        for j in range(m):#将数据分为训练集合测试集
            if (j < m * 0.9):
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(xArr, yArr)
        for k in range(30):
            matTestX = mat(testX);
            matTrainX = mat(trainX);
            meanTrainX = mean(matTrainX, 0)
            varTrainX = var(matTrainX, 0)
            #用训练是的参数将测试数据标准化
            matTestX = (matTestX - meanTrainX)/varTrainX

            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            errorMat[i,k] = resError(testY, yEst.T.A)
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[ nonzero(meanErrors == minMean)]
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0);
    varX = mean(xMat, 0)
    unReg = bestWeights/varX
    print 'the best model from Ridge Regression is:\n',unReg
    print 'with constant term: ', -1*sum(multiply(meanX, unReg)) + mean(yMat)

if __name__ == '__main__':
   # testStageWise()
   #  lgX = []; lgY = [];
   #  setDataCollect(lgX, lgY)
   #  print lgX
   #  print lgY
   arr = [[1,0,1,0], [0,1,0,1]]
   matrix = mat(arr)
   print matrix.nonzero()
   print matrix[matrix.nonzero()]
   print transpose(matrix.nonzero())



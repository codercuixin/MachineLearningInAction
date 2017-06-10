#!--**coding:utf-8**--
from numpy import *;

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.split('\t')
        fltLine = map(float, curline) #将curLine数组映射成float数组
        dataMat.append(fltLine)
    return dataMat
def bindSplitDataSet(dataSet, feature, value):
    '''
    二元切分
    按照feature所在的值进行二元切分
    dataSet[i,feature] > value就将第i行划分到mat0
    否则将第i行划分到mat1
    :param dataSet:
    :param feature:
    :param value:
    :return:
    '''
    mat0 = dataSet[nonzero(dataSet[:, feature]> value)[0], :]
    '''
    dataSet[:, feature]> value 返回True,False数组，将feature列中值大于value的置为True，其他的为False
    testMat = eye(4)
    print testMat
    print testMat[:, 1] > 0.5 #该列只有testMat[1][1]满足需求
    print nonzero(testMat[:, 1] > 0.5)
    print nonzero(testMat[:, 1] > 0.5)[0]
    print testMat[nonzero(testMat[:, 1] > 0.5)[0],:]
    print testMat[nonzero(testMat[:, 1] > 0.5)[0],:][0] #去一层[]

    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0.  1.]]
    [False  True False False]
    (array([1], dtype=int64),)
    [1]
    [[ 0.  1.  0.  0.]]
    [ 0.  1.  0.  0.]

    '''
    mat1 = dataSet[nonzero(dataSet[:, feature]<= value)[0], :]
    return mat0, mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])
def regErr(dataSet):
    #返回总方差
    return var(dataSet[:,-1]) * shape(dataSet)[0]
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    建立树
    :param dataSet:
    :param leafType: 表示建立叶节点的函数
    :param errType: 代表误差计算函数
    :param ops:
    :return:
    '''
    #找到最好的特征的下标，阈值
    feat ,val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat;
    retTree['spVal'] = val;
    #找到左子树的元素，右子树的元素
    lSet, rSet = bindSplitDataSet(dataSet, feat,val)
    #递归创建左子树，右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''
    选择用来划分数据最好特征和阈值

    对每个特征
        对每个特征值
            将数据集分成两份
            计算切分的误差
            如果当前误差小于最好误差，则将当前切分设为最切切分，并更新最好误差
    返回最佳切分的特征和阈值
    :param dataSet:
    :param leafType:负责生成叶节点，在回归树中就是目标变量的均值
    :param errType:误差函数
    :param ops:
    :return:
    '''
    #tolS为容忍的误差下降值，tolN为容忍的切分最小样本数
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        #如果所有数值都相同，就返回
        return None, leafType(dataSet)
    m,n = shape(dataSet)

    S = errType(dataSet)
    #最佳特征的选择是通过减少误差的平均值来驱动的
    bestS = inf; bestIndex = 0; bestValue = 9;
    for featIndex in range(n-1):
        for splitVal in dataSet[:,featIndex]:
            mat0, mat1 = bindSplitDataSet(dataSet, featIndex, splitVal)
            if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            #计算当前误差
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex;
                bestValue = splitVal
                bestS = newS

    #如果减少（S-bestS）小于阈值，则不进行分割
    if(S - bestS) < tolS:
        return None, leafType(dataSet)

    mat0, mat1 = bindSplitDataSet(dataSet, bestIndex, bestValue)
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): #如果两个子集小于容忍的最小样本数，也返回
          return None, leafType(dataSet)

    return bestIndex, bestValue

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right'])/2

def prune(tree, testData):
    '''
    对树进行后剪枝

    基于已有的数切分测试数据
        如果存在任一子集是一棵树，则在该子树递归剪枝过程
        计算当前两个节点合并后的误差
        计算不合并的误差
        如果合并的误差小于不合并的，那么就将两个叶节点合并
    :param tree:
    :param testData:
    :return:
    '''
    if shape(testData)[0] == 0 : return getMean(tree)
    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lSet ,rSet = bindSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = bindSplitDataSet(testData, tree['spInd'], tree['spVal'])

        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2))+\
            sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print 'merging'
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    '''
    求解weightBest并返回，以及X和Y
    :param dataSet:
    :return:
    '''
    m, n = shape(dataSet)
    #将X，Y格式化
    X = mat(ones((m, n)));
    Y = mat(ones((m, 1)));
    X[:, 1:n] = dataSet[:, 0:n-1]; #将dataSetSize的前n-1列放到X的从第二列开始的n-1列
    Y =dataSet[:, -1]
    XTX = X.T * X
    if linalg.det(XTX) == 0.0:
        raise NameError('This Matrix is singular, cannot do inverse\n');
        return;
    ws = XTX.I * (X.T * Y)
    return ws, X, Y;

def modelLeaf(dataSet):
    wS, X, Y = linearSolve(dataSet)
    return wS

def modelErr(dataSet):

    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    #返回方差
    return sum(power(yHat - Y, 2))
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    '''
    对回归树的叶节点进行预测
    :param model:
    :param inDat:
    :return:
    '''
    n = shape(inDat)[1]
    #对输入进行格式化处理，在原数据的基础上增加第0行
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    #计算并返回预测值
    return float(X * model)

def treeForecast(tree, inData, modelEval=regTreeEval,):
    '''
    对于输入的单个数据点或者行向量，函数会返回一个浮点值
    :param tree:
    :param inData:
    :param modelEval:
    :return:
    '''
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForecast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForecast(tree, testData, modelEval=regTreeEval):
    '''
    以向量的形式返回一组预测值
    :param tree:
    :param testData:
    :param modelEval:
    :return:
    '''
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForecast(tree, mat(testData[i]), modelEval)
    return yHat

def test():
     # myDat = loadDataSet('ex00.txt');
    # myDat = mat(myDat)
    # print createTree(myDat)
    #
    # myDat1 = loadDataSet('ex0.txt')
    # myDat1 = mat(myDat1)
    # print createTree(myDat1)
    #
    # myDat2 = loadDataSet('ex2.txt')
    # myDat2 = mat(myDat2)
    # myTree = createTree(myDat2, ops=(0, 1))
    # print myTree
    # #因为myDat2中的数值比较大，容忍的误差也放大一点
    # print createTree(myDat2, ops=(10000, 4))
    #
    # #测试后剪枝函数
    # myDat2Test = loadDataSet('ex2test.txt')
    # myDat2Test = mat(myDat2Test)
    # prune(myTree, myDat2Test)

    #测试模型树函数
    myMat2 = mat(loadDataSet('exp2.txt'))
    print createTree(myMat2, modelLeaf, modelErr, (1, 10))

def compareRegression():
    '''比较几个树回归和线性回归那个更好'''
    #建立回归树
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForecast(myTree, testMat[:, 0])
    print corrcoef(yHat, testMat[:, 1], rowvar=0)

    #建立模型树
    myTree = createTree(trainMat, modelLeaf, modelErr, (1,20))
    yHat = createForecast(myTree, testMat[:, 0], modelTreeEval)
    print corrcoef(yHat, testMat[:, 1],rowvar=0)

    #建立线性回归
    ws, X ,Y = linearSolve(trainMat)
    print ws
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i,0] * ws[1,0] + ws[0, 0]
    print corrcoef(yHat, testMat[:, 1],rowvar=0)
if __name__ == '__main__':

    compareRegression()

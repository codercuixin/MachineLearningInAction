# -*-coding:utf-8-*-
from math import  log
import operator
def calcShannonEnt(dataSet):
    '''
    计算dataset二维数组香农熵
    :param dataSet:
    :return:
    '''
    length = len(dataSet)
    lableCounts = {} #dictionary
    for featVal in dataSet:
        currentLable = featVal[-1]
        if currentLable not in lableCounts.keys():
            lableCounts[currentLable] = 0;
        lableCounts[currentLable] += 1.0
    shnnoent = 0
    for key in lableCounts.keys():
        prob = (float)(lableCounts[key]) /length
        shnnoent -= prob*log(prob,2)
    return shnnoent
def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['no surfacing' , 'flippers']
    return dataSet ,labels


def splitDataSet(dataSet, axis, value):
    '''
    对于dataSet中的每一行，该行特征下标axis，值等于value的加入子矩阵。
    :param dataSet:待划分的数据集
    :param axis:轴，划分数据集的特征下标
    :param value:要等于value才可以返回
    :return: 返回满足要求的子矩阵。
    '''
    retDataSet = [];
    for featVec in dataSet:
        if featVec[axis] == value:
            #前闭后开，为了过滤划分元素
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            #为了让其形成二维的
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureIndexToSplit(dataSet):
    '''
    使用所有没有使用的属性并计算与之相关的样本熵(无序度）值
    选取其中熵值最小的属性
    :param dataSet: 应该是二维列表（矩阵）；每一行最后一个为标签
    :return: 返回选择出来的最好属性的下标
    '''
    numOfFeature = len(dataSet[0]) - 1;
    baseEntropy = calcShannonEnt(dataSet);
    bestInfoGain = 0.0;
    bestFeature = -1 ;
    for i in range(numOfFeature):
        featList = [example[i] for example in dataSet];
        uniqueVals = set(featList);
        newEntropy = 0.0;
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #信息增益越大即熵（数据无序度）的减少
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i;
    return bestFeature

def majority_count(classList):
    #定义一个字典，键为标签，值为标签出现的次数
    class_count={}
    for count in classList:
        if(count not in class_count.keys()):
            class_count[count] = 0;
        class_count[count] += 1
    #将class_count按照出现的频率由大到小进行排序
    sortedClassCount = sorted(class_count.iteritems(),
                              key=operator.itemgetter(1)
                              ,reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, featureLabels):
    '''
    利用ID3算法创建树 https://zh.wikipedia.org/wiki/ID3算法
    这个算法是建立在奥卡姆剃刀的基础上：越是小型的决策树越优于大的决策树（简单理论）。
    尽管如此，该算法也不是总是生成最小的树形结构。而是一个启发式算法。奥卡姆剃刀阐述了一个信息熵的概念：
    这个ID3算法可以归纳为以下几点：
    使用所有没有使用的属性并计算与之相关的样本熵值
    选取其中熵值最小的属性
    生成包含该属性的节点
    :param dataSet:
    :param featureLabels:
    :return:
    '''
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList) :
        return classList[0]
    #遍历完所有特征之后，返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majority_count(classList)
    bestFeatureIndex = chooseBestFeatureIndexToSplit(dataSet)
    bestFeatureLabel = featureLabels[bestFeatureIndex]
    myTree = {bestFeatureLabel:{}}
    #将这个元素从lables中删除
    del(featureLabels[bestFeatureIndex])
    #得到这一特征的所有值，也就是矩阵的一列
    featureValues = [example[bestFeatureIndex] for example in dataSet]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        #拷贝featureLabels到subLabels
        subLabels = featureLabels[:]
        childTree = createTree(splitDataSet(dataSet, bestFeatureIndex, value)
                       , subLabels);
        myTree[bestFeatureLabel][value] = childTree;
    return myTree

def classify(inputTree, featLabels, testVec):
    """

    :param inputTree:  a tree consist of features,has-feature-or-not,the classify result
    :param featLabels: features list
    :param testVec: test data to judge the function is right or not
    :return:
    """
    first_feature = inputTree.keys()[0]
    secondDict = inputTree[first_feature]
    #找到当前判断的特征在特征列表中所处的下标
    feature_index = featLabels.index(first_feature)
    current_feature_value = testVec[feature_index] #测试数据中是否具有某个特征

    for key in secondDict.keys():
        if current_feature_value == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel
def storeTree(inputTree, fileName):
    """
    使用Pickle模块来保存序列化对象
    :return:
    """
    import pickle
    fw = open(fileName,'w')
    pickle.dump(inputTree, fw)
    fw.close()
def grabTree(fileName):
    '''
    恢复inputTree字典
    :param fileName:
    :return:
    '''
    import pickle
    fr = open(fileName, "r")
    inputTree = pickle.load(fr)
    fr.close()
    return inputTree
def loadAndShowLense():
    import treePlotter
    fr = open("lenses.txt", 'r')
    lenses = [ lineString.strip().split('\t') for lineString in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    print lensesTree
    treePlotter.createPlot(lensesTree)
if __name__ == '__main__':

    # print chooseBestFeatureToSplit(dataSet)
    # returnVec = splitDataSet(dataSet,0,1)
    # print returnVec
    # returnVec = splitDataSet(dataSet,0,0)
    # print returnVec
    # shannonEnt = calcShannonEnt(dataSet)
    # print shannonEnt
    # dataSet[0][-1] = 'maybe'
    # shannonEnt = calcShannonEnt(dataSet)
    # print shannonEnt

    # import treePlotter
    # myDat, labels = createDataSet()
    # # print labels
    # myTree = treePlotter.retriveTree(0)
    # # print myTree
    # # print classify(myTree, labels, [1,0])
    # # print classify(myTree, labels, [1,1])
    # # print classify(myTree, labels, [0,1,1,1]) #first 0 decide the type
    # storeTree(myTree, 'mytree.txt')
    # print grabTree('mytree.txt')

    loadAndShowLense()

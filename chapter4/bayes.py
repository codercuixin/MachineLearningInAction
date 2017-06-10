# *--coding:utf-8--*
from numpy import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] # 0 代表侮辱性言语，1代表正常言论
    return postingList,classVec
def createVocabList(dataSet):
    '''
    将dataSet中的不同元素组成一个list
    :param dataSet:
    :return:
    '''
    vocabList = set([]) #创建一个空集来存储
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)
def setOfWords2Vec(vocubList, inputSet):
    '''
    将输入的单词们转化成向量。创建一个与vocubList同长的全0列表returnList，如果inputSet中出现在vocubList里，就置同等位置的为1
    :param vocubList:
    :param inputSet:
    :return:
    '''
    returnVec = [0] * len(vocubList)
    for inputWord in inputSet:
        if inputWord in vocubList :
            index = vocubList.index(inputWord)
            returnVec[index] = 1
        else:
            print 'The word :%s is not in vocabulary!' %inputWord
    return  returnVec
def bagOfWords2VecMN(vocubList, inputSet):
    '''
    将输入的单词set转化成一个向量
    向量每个位置表示该单词出现的次数。
    :param vocubList:
    :param inputSet:
    :return:
    '''
    returnVec = [0] * len(vocubList)
    for inputWord in inputSet:
        if inputWord in vocubList:
            returnVec[ vocubList.index(inputWord)] += 1.0
    return returnVec
def trainNBO (trainMatrix, trainCategory):
    '''
    获取条件概率p(w\ci), 其中ci为类别，w为对应的向量。
    获取类别概率p(ci),其中ci为类别。
    :param trainMatrix: 文档矩阵
    :param trainCategory: 每篇文档所属类别组成的向量，1表示侮辱，0表示正常
    :return:
    '''

    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Nums = ones(numWords)
    p1Nums = ones(numWords)
    p0Denom = 2.0 #计数器,解决0值问题
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Nums += trainMatrix[i] #矩阵相加
            p1Denom += sum(trainMatrix[i])
        else:
            p0Nums += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vec = log(p1Nums/p1Denom) #给每个元素做除法，
    p0Vec = log(p0Nums/p0Denom) #p1Vec,p0Vec 刚好是各自的相对改了P(w|ci)
    # print p1Vec
    # print p1Nums
    # print p1Denom
    return p0Vec, p1Vec, pAbusive
def classifyNB(vec2classify, p0Vec, p1Vec, pClass1):
    '''

    :param vec2classify: 要分类的向量vex2Classify
    :param p0Vec:  使用函数trainNBO返回的三个参数。
    :param p1Vec:
    :param pClass1:
    :return:
    '''
    p1 = sum(vec2classify * p1Vec)  + log(pClass1)
    p0 = sum(vec2classify * p0Vec) + log(1- pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOfPosts, listOfClasses = loadDataSet()
    myVocbaList = createVocabList(listOfPosts)
    trainMatrix = []
    for post in listOfPosts:
        trainMatrix.append(setOfWords2Vec(myVocbaList, post))
    p0Vec, p1Vec, pAbusive = trainNBO(trainMatrix, listOfClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocbaList, testEntry))
    print testEntry,'classfied as: ',classifyNB(thisDoc, p0Vec, p1Vec, pAbusive)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocbaList, testEntry))
    print testEntry,'classfied as: ',classifyNB(thisDoc, p0Vec, p1Vec, pAbusive)

def textParse(bigString):
    '''
    将大的字符串解析后才能列表
    :param bigString:
    :return:
    '''
    import  re
    listOfTokens = re.split(r'\W*',bigString)
    return [ token.lower() for token in listOfTokens if len(token) > 2]

def spamTest():
    docList = [];
    classList = [];
    fullText = [];
    for i in range(1,26):#前闭后开
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)#形成矩阵
        fullText.extend(wordList)#还是一维数组
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabularyList = createVocabList(docList)
    trainningSet = range(50)
    # print trainningSet
    testSet = []
    for i in range(10):
        randomIndex = int( random.uniform(0, len(trainningSet)))
        testSet.append(trainningSet[randomIndex])
        # print randomIndex,trainningSet[randomIndex]
        del(trainningSet[randomIndex])
    trainningMatrix = [];trainningClasses = [] ;
    for docIndex in trainningSet:
        trainningMatrix.append(setOfWords2Vec(vocabularyList, docList[docIndex]))
        trainningClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNBO(trainningMatrix, trainningClasses)
    errorCount = 0.0;
    for docIndex in testSet:
        wordVec = setOfWords2Vec(vocabularyList, docList[docIndex])
        if (classifyNB(wordVec, p0V, p1V, pSpam)) != classList[docIndex]:
            errorCount += 1.0
    print "the error rate is ",float(errorCount)/len(testSet)
    return float(errorCount)/len(testSet)
def calcMostFreq(vocabList, fullText):
    '''
    返回在词汇表出现频率最高的前30个
    :param vocabList:
    :param fullText:
    :return:
    '''
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(),
                        key=operator.itemgetter(1),\
                        reverse=True
                        )
    # print sortedFreq[:30]
    return sortedFreq[:30] #只返回前30个的



def localWords(feed1, feed0):
    docList= []; classList = []; fullText = [];
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList) #矩阵
        fullText.extend(wordList) #列表
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    print 'orignal vocablist: ',vocabList
    top30List = calcMostFreq(vocabList, fullText)
    print 'top 30 ;',top30List
    for topRateWord in top30List:
        if topRateWord[0] in vocabList:
            vocabList.remove(topRateWord[0])
    print 'delete top30: ',vocabList
    trainingSet = range(2*minLen)
    testSet = [];
    for i in range(20):
        randomoIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randomoIndex])
        del(trainingSet[randomoIndex])
    trainingMat = [] ;
    trainingClasses = []
    for docIndex in trainingSet:
        trainingMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainingClasses.append(classList[docIndex])
    p0Vector,p1Vector,pSpam =trainNBO(array(trainingMat), array(trainingClasses))

    #下面测试错误率
    errorCount = 0.0;
    for docIndex in testSet:
        if classifyNB(bagOfWords2VecMN(vocabList,docList[docIndex]), p0Vector, p1Vector, pSpam )\
            != classList[docIndex]:
            errorCount += 1
    print "the error rate is ",float(errorCount)/len(testSet)
    return vocabList, p0Vector, p1Vector

def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = [] ; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key= lambda pair:pair[1], reverse=True)
    print 'SP**SP**SP**SP**SP**SP**SP**SP**SP**SP**SP**SP**SP**SP**'
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key= lambda pair:pair[1], reverse=True)
    print 'NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**'
    for item in sortedNY:
        print item[0]

if __name__ == '__main__':
    # testingNB()
    # averageErrorCount = 0.0;
    # for i in range(10):
    #     averageErrorCount += spamTest()
    # print "the average error rate is ",averageErrorCount/10
    # # fw = open('craigslist.txt','w');
    # #
    # # fw.write(ny); # expected a string or other character buffer object
    # # fw.close()
    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny,sf)
    # print ny
    # print ny['entries']
    # print len(ny['entries'])
    # print sf
    # print sf['entries']
    # print len(sf['entries'])
    # print '\n\n'
    # vocabList, pSF, pNF = localWords(ny, sf)
    #
    # v0cabList, pSF, pNF = localWords(ny, sf)




#coding:utf-8
class treeNode:
    '''
    FP数的类定义
    '''
    def __init__(self, nameValue, numOccur, parentNode):
         self.name = nameValue
         self.count = numOccur
         self.nodeLink = None
         self.parent = parentNode
         self.children = {}
    def inc(self, numOccur):
        self.count += numOccur

    def display(self, ind=1):
        print ' '*ind, self.name, ' ',self.count
        for child in self.children.values():
            child.display(ind+1)
def createTree(dataSet, minSup=1):
    '''
    穿件一个FPTree
    :param dataSet: 是一个字典，键为一组交易数组，值为出现的次数。
    例如{frozenset(['e', 'm', 'q', 's', 't', 'y', 'x', 'z']): 1, frozenset(['x', 's', 'r', 'o', 'n']): 1,
     frozenset(['s', 'u', 't', 'w', 'v', 'y', 'x', 'z']): 1,frozenset(['q', 'p', 'r', 't', 'y', 'x', 'z']): 1,
     frozenset(['h', 'r', 'z', 'p', 'j']): 1, frozenset(['z']): 1}
    :param minSup: 最小支持度，这里用现的次数来表示
    :return:
    '''
    headerTable = {} #头指针表
    #扫描数据集并统计每个元素项出现的频度，将这些信息保存在头指针表中
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans] #dataSet[trans]表示该交易出现的次数

    #删除不满足最小支持度的元素项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) ==0 : return None, None; #如果没有元素项就返回

    for k in headerTable:
        #对头指针表加以扩展，在保存出现次数的基础上，可以保存指向每种类型第一个元素的指针
        headerTable[k] = [headerTable[k], None]

    #创建一个根节点
    retTree = treeNode('Null Set', 1, None)
    #再一次遍历数据集，这次只考虑频繁项
    for transSet, count in dataSet.items():
        localD = {} #获得每一行数据中每一项数据在总体数据中出现的次数，以此来排序
        for item in transSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD)>0:
            #利用一行数据在总体数据中出现的次数从高到低进行排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            #使用排序后的一行数据对树进行填充
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable;
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children: #如果items[0]是当前树根节点的子节点，那么直接增加值即可
        inTree.children[items[0]].inc(count)
    else:#如果在当前树的子节点中不存在items[0]， 那么就将items[0]插到子节点中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)

        if headerTable[items[0]][1] == None:#头结点链表记录下指向每种类型第一个元素的指针
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items)> 1:
        #将剩余的节点，也插入到树中，只不过他们的父节点变为了刚刚加入的items[0]
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(head, targetNode):
    '''
    将targetNode插入到链表的最后一个
    :param head:
    :param targetNode:
    :return:
    '''
    while(head.nodeLink != None):
        head = head.nodeLink
    head.nodeLink = targetNode
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        #迭代上溯直到根节点
        ascendTree(leafNode.parent, prefixPath)
def findPrefixPath(basePat, treeNode):
    '''
    发现以给定元素项结尾的所有路径的函数
    :param basePat:
    :param treeNode:
    :return:
    '''
    condPats = {} #条件模式基字典，键前缀树，值为起始元素的计数值
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count #键前缀树，值为起始元素的计数值
            #因为要作为map的key，所以加一个frozenset
        treeNode = treeNode.nodeLink
    return condPats;


def loadSimpleData():
    simpleData = [['r', 'z','h', 'j', 'p'],
                  ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                  ['z'],
                  ['r', 'x', 'n', 'o', 's'],
                  ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                  ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
                  ]
    return simpleData;
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1 #后面为其出现的次数
    return retDict;

def mineTree(inTree, headerTable, minSup, prefix, freqItemList):
    #对头指针表中的元素项按照其出现频率进行从小到大排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1])]

    for basePat in bigL:
        newFreqSet = prefix.copy();
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        #创建条件基
        condPatBases = findPrefixPath(basePat, headerTable[basePat][1])
        #利用条件基穿件条件树
        myCondTree, myHead = createTree(condPatBases, minSup)
        if myHead != None: #如果树中有元素的话，递归调用mineTree
            print 'conditional tree for: ',newFreqSet
            myCondTree.display()
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
import twitter
from time import sleep
import re
def getLotsOfTweets(searchStr):
    CONSUMER_KEY = 's4VEmUKxeyqt01ECqTEY55Zsm'
    CONSUMER_SECERET = 'jhIdxg6NWB37d0a3gUuUAkyDXEKaswLhDaXreWy8zGVsMmVlxt'
    ACCESS_TOKEN_KEY = '838940044240965633-3bNYUX9a3qToytK0HnPDuVor3DuM5dD'
    ACCESS_TOKEN_SECERT = '7nHtyrCcszdBeStarzaiyEqfEIH8CJ2GUkaVZCxH98lTi'
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECERET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECERT)
    # print api.VerifyCredentials()
    # print api._GetFriendsFollowers()
    # 每一百条推文作为一页
    resultPages = []
    for i in range(1, 15):
        print 'fetching page %d'%i
        searchResults = api.get(searchStr, per_page=1, page=1)
        resultPages.append(searchResults)
        sleep(6)
    return resultPages
def textParse(bigString):
    #去掉url
    urlsRemoved = re.sub('(http[s]?:[/][/]|www.)([a-z][A-Z][0-9]|[/.]|[~]*)', '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [ tok.lower() for tok in listOfTokens if len(tok) > 2 ]
def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList


if __name__ == '__main__':
    #测试TreeNode
    # rootNode = treeNode('pyramid', 9, None)
    # rootNode.children['eye']= treeNode('eye', 13, None)
    # rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    # rootNode.display()
    # simpleData = loadSimpleData();
    # # print simpleData
    # initSet =  createInitSet(simpleData)
    # # print initSet
    # myFPTree, myHeadTable = createTree(initSet, 3)
    # myFPTree.display()
    # print myHeadTable

    # print findPrefixPath('x', myHeadTable['x'][1]) # myHeadTable['x'][1]指向的是头结点
    # print findPrefixPath('z', myHeadTable['z'][1])
    # print findPrefixPath('r', myHeadTable['r'][1])

    #测试mineTree条件树函数
    # freqItemList = []
    # mineTree(myFPTree, myHeadTable, 3, set([]), freqItemList)
    # print freqItemList

    #测试获取推文函数
    # lotsOfTwitters =getLotsOfTweets('RIMM')
    # listOfTerms = mineTweets(lotsOfTwitters, 20)
    # print len(listOfTerms)
    # print listOfTerms

    #查看用户浏览的频繁项集
    parseDat = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parseDat)
    myFPTree, myHeadTable = createTree(initSet, 100000)
    myFreqList =[]
    mineTree(myFPTree, myHeadTable, 100000, set([]), myFreqList)
    print len(myFreqList)
    print myFreqList

# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
#定义箭头和文本格式
decisionNode = dict(boxstyle='sawtooth', fc= '0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

#绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center',ha='center', bbox=nodeType,
                            arrowprops=arrow_args)
def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,
     plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
# def createPlot(inTree):
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     axprops = dict(xticks=[],yticks=[])
#     #定义一个绘图区域
#     createPlot.ax1 = plt.subplot(111,frameon=False, **axprops)
#     #下面两个全局变量存储书的宽度和高度
#     plotTree.totalW = (float)(getNumLeafs(inTree))
#     plotTree.totalD = (float)(getTreeDepth(inTree))
#     #下面两个全局变量追踪已经绘制的节点位置，以及下一个节点的恰当位置。
#     plotTree.xOff = -0.5/plotTree.totalW;
#     plotTree.yOff = 1.0;
#     plotTree(inTree, (0.5, 1.0), "")
#     plt.show()
#获得一颗树（用字典表示）的叶子结点的个数
#如果遇到的key是dict，那么就继续遍历
#如果遇到的key类型不是dictionary，那么就表明是叶子结点直接加一即可。
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs
#获取一棵树（字典）的深度
#使用一个循环遍历所有的根节点的子节点，
#当面对这个子节点时，如果他是字典，则遍历这个字典得到他的深度；最后加一
#当面对的这个子节点不是字典时，则这个字数深度为1
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth>maxDepth:
            maxDepth = thisDepth
    return  maxDepth
#在父子元素之间添加文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
def retriveTree(i):
    listOfTrees = [{'no surfacing':{0:'no',
                                     1:{'flippers':{0:'no', 1:'yes'}}}}
                   ,{'no surfacing':{0:'no', 1:{"flippers":
                    {0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
                   ]
    return listOfTrees[i]
if __name__ == '__main__':
    # createPlot();
    # print retriveTree(0)
    # print retriveTree(1)
    # print getNumLeafs(retriveTree(0))
    # print getTreeDepth(retriveTree(0))

    myTree = retriveTree(0)
    createPlot(myTree)
    myTree['no surfacing'][3] = 'maybe'
    createPlot(myTree)


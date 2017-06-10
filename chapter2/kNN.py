from numpy import  *
import matplotlib
import matplotlib.pyplot as plot
import operator
def createDataSet():
    group = array([[1.0 ,1.1] , [1.0 , 1.0] ,[0 , 0] , [0 , 0.1]])
    labels = ['A','A','B','B']
    return group , labels
# first find the  k data in dataSet which is neareast to inX
#second compute the lables which occur most in these k data
def classify0( inX ,dataSet , labels,k):
    #get the row of the dataSet matrix
    datasetSize= dataSet.shape[0];
    #tile copy the inX into dataSetSize rows,then minus the dataSet
    diffMat = tile(inX , (datasetSize ,1)) - dataSet
    sqDiffMat = diffMat**2;
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistIndicies =distance.argsort();
    classCount ={}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def file2matrix(filename):
    fr = open(filename)
    arrayLines= fr.readlines()
    numberOfLines= len(arrayLines)
    returnMatrix =zeros((numberOfLines ,3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip();
        listFromLines = line.split('\t')
        returnMatrix[index:] = listFromLines[0:3]
        classLabelVector.append((int)(listFromLines[-1]))
        index +=1
    return returnMatrix ,classLabelVector
#change the oldData into 0~1
def autoNorm(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue -minValue
    length = dataSet.shape[0]
    normDataSet = dataSet-tile(minValue,(length,1))
    normDataSet = normDataSet/tile(ranges,(length,1))
    return normDataSet ,ranges,minValue

def datingClassTest():
    hoRatio = 0.10
    datingDataMatrix,datingDataLables = file2matrix('datingTestSet2.txt')
    normat, ranges , minValue = autoNorm(datingDataMatrix)
    m = normat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify0(normat[i,:], normat[numTestVecs:m,:],
                                    datingDataLables[numTestVecs:m] , 3)
        print('the classifier came back with:%d,the real answer is:%d'%(classfierResult , datingDataLables[i]))
        if(classfierResult !=datingDataLables[i]): errorCount +=1
    print('the total error rate is %f'%(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of spent playing video games?"))
    ffMiles = float(raw_input("frequent flies miles earned in a year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMatrix,datingDataLables = file2matrix("datingTestSet2.txt")
    normat, ranges , minValue = autoNorm(datingDataMatrix)
    figure = plot.figure()
    ax = figure.add_subplot(111)
    ax.scatter(datingDataMatrix[:, 0], datingDataMatrix[:,1],15.0*array(datingDataLables),15.0*array(datingDataLables))
    plot.show()
    print datingDataLables
    print datingDataMatrix
    inArr = array([ffMiles, percentTats, iceCream])
    classifyResult = classify0(inArr-minValue/ranges, normat , datingDataLables, 3)
    print ("you will probably like this person:",resultList[classifyResult-1])
if __name__ == '__main__':
    classifyPerson()
# datingClassTest()
# group,labels =createDataSet();
#  print group
# print labels
# print classify0([0,0] , group ,labels , 3)

# datingDataMatrix , datingDataLables=file2matrix('datingTestSet2.txt')
# normDataSet , ranges ,minValue= autoNorm(datingDataMatrix)
# print normDataSet

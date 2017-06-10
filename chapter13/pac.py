#coding:utf-8
from numpy import *
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # print 'stringArr: ', stringArr
    dataArr = [map(float, line) for line in stringArr]
    # print 'dataArr: ', dataArr
    return mat(dataArr)
def pca(dataMat, topNfeat =9999999):
    '''
    主成分分析法，principal component analysis
    将数据转换成前topNfeat个主成分

    :param dataMat: 数据集
    :param topNfeat:主成分个数
    :return:
    '''
    meanVals = mean(dataMat, axis=0) #按照列的方向取均值 得到1*n的行向量
    # print 'meanVals: ',meanVals
    meanRemoved = dataMat - meanVals #与该均值行向量作差
    # print 'meanRemvoved: ', meanRemoved
    covMat = cov(meanRemoved, rowvar=0) #计算协方差矩阵
    # print 'covmat: ', covMat
    eigVals, eigVects = linalg.eig(mat(covMat)) #计算协方差矩阵的特征值和特征向量
    # print 'eigVals: ', eigVals
    # print 'eigVects: ', eigVects
    eigValInd = argsort(eigVals) #返回从小到大访问eigVals值的下标数组
    eigValInd = eigValInd[:-(topNfeat +1):-1] #倒着取topNfeat+1个，其实这里是将选择前topNfeat个大的特征值的下标
    redEigVects = eigVects[:, eigValInd] #将对应大的特征向量按照列的方式取出来，
    # print 'redEigVects: ', redEigVects

    #将原始数据转换到新空间
    lowDimensionDataMat = meanRemoved * redEigVects # (m, n) * (n, numFeature)= (m, numFeature)
    # print 'lowDimensionDataMat: ', lowDimensionDataMat
    #重构原始数据
    reconMat = (lowDimensionDataMat * redEigVects.T) + meanVals #(m,numFeature) *(numFeature, n) + (m, n) = (m, n)
    # print 'reconMat: ', reconMat.T
    return lowDimensionDataMat, reconMat

def showData(dataMat, reconMat):
    import  matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()
# def replaceNanWithMean():
#     '''
#     加载数据，将一列的nan(not a number)替换为该列所有非nan元素的均值
#     :return:
#     '''
#     dataMat = loadDataSet('secom.data', ' ')
#     numFeat = shape(dataMat)[1]
#
#     for i in range(numFeat):
#         #获取第i列中不为nan的数的均值
#         meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i]);
#         #用该均值替换该列中的每一个nan
#         dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal;
#     return dataMat
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    print 'numFeat', numFeat
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
def computevar(reconMat, dataMat):
    deltaMat = dataMat - reconMat
    return var(deltaMat)

if __name__ == '__main__':
    # dataMat = loadDataSet('testSet.txt')
    # lowDMat, reconMat = pca(dataMat, 1)
    # print shape(lowDMat)
    # showData(dataMat,reconMat)
    # lowDMat, reconMat = pca(dataMat, 2)
    # print shape(lowDMat)
    # showData(dataMat, reconMat)

    dataMat = replaceNanWithMean();
    # print dataMat
    meanVal = mean(dataMat, axis=0) #按列取均值
    meanRemoved = dataMat - meanVal;
    covMat = cov(meanRemoved, rowvar=0) #计算去除均值的协方差
    eigVals, eigVects = linalg.eig(mat(covMat)) #计算特征值和特征向量
    # print eigVals
    # print len(eigVals)#590
    # print len(eigVals[nonzero(eigVals == 0.0 )])#116
    # lowDMat, reconMat = pca(dataMat, 6)
    # print lowDMat
    # print reconMat

    arr = []
    sum =0.0
    for i in range(0, 20): #我只算了前20个，你可以将20换成590
        lowDMat, reconMat = pca(dataMat, i)
        varI = computevar(reconMat, dataMat)
        sum += varI
        arr.append(varI)
    print sum;
    print arr;
    sumPercent = 0.0
    for i in range(0, 20):
        thisPercent = arr[i]/sum;
        sumPercent += thisPercent;
        print 'pac %d: this takes %f, all takes %f '%(i, thisPercent, sumPercent)












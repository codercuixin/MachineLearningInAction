#!--**coding:utf-8**--
from numpy import *

def loadDataSet(fileName):
    file = open(fileName);
    dataSet = []
    for line in file.readlines():
        curline = line.split('\t')
        floatline = map(float, curline)
        dataSet.append(floatline)
    return dataSet;

def distanceEuclidean(vecA, vecB):
    '''
    计算nvecA, vecB之间的欧式距离，
    先对应位置相减，然后对每个元素求平方，然后对所有元素求和，最后开方
    :param vecA:
    :param vecB:
    :return:
    '''
    return sqrt(sum(power(vecA- vecB, 2 )))

def randCentroids(dataSet, k):
    '''
    生成k个随机质心
    思想：
    生成k*n的零矩阵centroids
    对于每一列j
        将该列所有值置为随机值，该值要处在数据集对应列最小值，最大值之间
    返回centroids
    :param dataSet:
    :param k:
    :return:
    '''
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])#数据中j列最小值
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ) #需要强制装换一下，要不然是1*1的，与下面的random.rand(k, 1)会产生冲突
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))#random.rand(k, 1) 返回k*1的ndarray,每一个都是0-1随机值
    return centroids
def kMeans(dataSet, k, distMeasure=distanceEuclidean, createCentroids=randCentroids):
    '''
    K-均值聚类算法以k个随机质心开始。

    算法会计算每个点到质心的距离，每个点被分配到距离其最近的簇质心，
    然后紧接着基于新分配到簇的点更新簇质心
    以上过程重复数次，直到簇质心不再发生改变。
    :param dataSet:
    :param k:
    :param distMeasure:
    :param createCentroids:
    :return:
    '''
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#簇分配结果矩阵，记录每个点的簇分配结果。
    # 第一列记录簇索引值，第二列记录误差(指的是从当前点到簇质心的距离）

    centroids = createCentroids(dataSet, k)
    clusterChanged = True;
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):
            minDist = inf; minIndex = -1;#分别用来记录当前数据点与最近质心之间的距离，以及该最近质心的下标

            #下面开始寻找最近的质心
            for j in range(k):
                distJI = distMeasure(centroids[j, :], dataSet[i, :])
                if distJI<minDist:
                    minDist = distJI
                    minIndex = j
            #如果最小质心下标不等于原来的簇分类结果，就将clusterChanged置为TRUE
            if clusterAssment[i, 0] != minIndex: clusterChanged = True;
            clusterAssment[i, :] = minIndex, minDist**2
        print centroids

        #更新所有的质心
        for centroidIndex in range(k):
            #使用数组过滤的方式获得给定簇的所有点
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == centroidIndex)[0]]

            #clusterAssment[:, 0]为第一列，记录簇索引值；.A将其装换成ndarray;
            # clusterAssment[:, 0].A == centroidIndex然后将与当前centroidIndex相等的置为True,其他的置为False
            #nonzero(clusterAssment[:, 0].A == centroidIndex) 返回所有为True的下标ndarray

            #更新质心的值； axis=0按照列的方向对pointsInCluster求均值
            centroids[centroidIndex, :] = mean(pointsInCluster, axis=0)
    return centroids, clusterAssment
def testKMeans():
    # datMat = mat(loadDataSet('testSet2.txt'))
    #显示原始数据的分布
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datMat[:,0].A, datMat[:,1].A) #需要将matrix转换成ndarray
    # plt.show()

    # print min(datMat[:, 0])
    # print max(datMat[:, 0])
    # print min(datMat[:, 1])
    # print max(datMat[:, 1])
    # print randCentroids(datMat, 2)
    # print distanceEuclidean(datMat[0], datMat[1]) #第0个点和第1个点之间的距离

    # #测试数据一
    # datMat = mat(loadDataSet('testSet.txt'))
    # myCentroids,clusterAsseble =kMeans(datMat, 4)
    # #   显示原始数据及质心的分布
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datMat[:,0].A, datMat[:,1].A) #需要将matrix转换成ndarray
    # ax.scatter(myCentroids[:,0].A, myCentroids[:,1].A) #需要将matrix转换成ndarray
    # plt.show()


    #测试数据二
    datMat = mat(loadDataSet('testSet2.txt'))
    myCentroids,clusterAsseble =kMeans(datMat, 3)
    #   显示原始数据及质心的分布
    showKMeansResult(datMat, myCentroids)

def showKMeansResult(datMat, centroids):
    '''显示原始数据及质心的分布'''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datMat[:,0].A, datMat[:,1].A) #需要将matrix转换成ndarray
    ax.scatter(centroids[:,0].A, centroids[:,1].A) #需要将matrix转换成ndarray
    plt.show()

def biKMeans(dataSet, k, distanceMeasure=distanceEuclidean):
    '''
    二分K-均值算法
    首先将所有点作为一个簇，然后使用K-均值算法(k=2)对它进行划分
    下一次迭代时，选择有最大误差的簇划分。该过程重复直到k个簇创建完成
    :param dataSet:
    :param k:
    :param distanceMeasure:
    :return:
    '''
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2))) #第一列保存每个点的簇分配结果以及平方误差
    centroids0 = mean(dataSet, axis=0).tolist()[0] #计算数据集的质心
    centroidList = [centroids0] #保留所有的质心
    print centroidList
    for j in range(m):
        #计算每个点到初始质心之间的误差值
        clusterAssment[j, 1] = distanceMeasure(mat(centroids0), dataSet[j, :]) **2

    while(len(centroidList)<k):
        lowestSSE = inf
        for i in range(len(centroidList)):
            #找到每个质点对应的簇
            pointsInCurCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            #使用这一族数据尝试着生成两个质心
            centroidMat, splitClusterAss= \
                kMeans(pointsInCurCluster, 2, distanceMeasure);
            #计算如果划分这一族数据会产生的误差
            sseSplit = sum(splitClusterAss[:, 1])
            #计算不属于这一族即剩余数据的误差和
            sseNoSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print 'sseSplit and notSplit: ', sseSplit, sseNoSplit

            #如果划分误差加剩余数据误差之和小于当前最小误差，则保存本次划分
            if(sseSplit + sseNoSplit < lowestSSE):
                bestCentroidToSplit = i;
                bestNewCentroids = centroidMat
                bestClusterAss = splitClusterAss.copy() #会得到0和1两个结果簇
                lowestSSE = sseSplit + sseNoSplit
        #更新最好划分簇和新加簇的编号
        bestClusterAss[nonzero(bestClusterAss[:, 0].A == 1)[0], 0] = len(centroidList)
        bestClusterAss[nonzero(bestClusterAss[:, 0].A == 0)[0], 0] = bestCentroidToSplit

        print 'the bestCentroidToSplit is: ', bestCentroidToSplit
        print 'the len of bestClustAss is: ', len(bestClusterAss)
        #更新新的簇分配结果
        centroidList[bestCentroidToSplit] = bestNewCentroids[0, :].tolist()[0]
        # .tolist()[0]将行矩阵转换成list并且去掉外一层[],例如：matrix([[-0.45965615, -2.7782156 ]])转换成[-0.45965615, -2.7782156 ]
        centroidList.append(bestNewCentroids[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentroidToSplit)[0], :] = bestClusterAss
    print centroidList
    return mat(centroidList), clusterAssment
def testBiKMeans():
    dataMat3 = mat(loadDataSet('testSet2.txt'))
    # biKMeans(dataMat3, 5)
    # centroidList, myNewAssments = biKMeans(dataMat3, 3)
    centroidList, myNewAssments = biKMeans(dataMat3, 4)
    # centroidList, myNewAssments = biKMeans(dataMat3, 5)
    print centroidList
    print myNewAssments
    showKMeansResult(dataMat3, centroidList)

import urllib
import json
def geoGrab(stAddress, city):
    '''
    获取geo信息，然而下面的 API已经不管用了
    :param stAddress:
    :param city:
    :return: 返回一个字典
    '''
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print yahooApi
    c=urllib.urlopen(yahooApi)
    #将json数据转换成Python对应的数据格式
    return json.loads(c.read())

from time import  sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2] )
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            fw.write('%s\t%f\t%f\t' %(line, lat, lng))
        else: print 'error fetching'
        sleep(1) #不要频繁调用API，以防被封禁
    fw.close()

def distanceGreatCircleDistance(vecA, vecB):
    '''
    求两个球面(这里是地球）上点的距离
    https://zh.wikipedia.org/wiki/%E5%A4%A7%E5%9C%86%E8%B7%9D%E7%A6%BB
    http://baike.baidu.com/item/%E5%A4%A7%E5%9C%86%E8%B7%9D%E7%A6%BB
    :param vecA:
    :param vecB:
    :return:
    '''
    a = sin(vecA[0, 1]*pi/180) * sin(vecB[0, 1]*pi/180) # *pi/180将经纬度从角度装换成弧度
    b = cos(vecA[0, 1]*pi/180) * cos(vecB[0, 1]*pi/180)*cos(pi*(vecB[0,0] - vecA[0,0])/180)
     #6371.0是地球平均半径
    return arccos(a+b) * 6371.0
import  matplotlib
import  matplotlib.pyplot as plt;
def clusterClubs(numCluster=5):

    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])]) #分别对应着经度，纬度
    datMat = mat(datList)
    myCentroids, clusterAssing = biKMeans(datMat, numCluster, distanceGreatCircleDistance)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png') #读图片， 基于一份图像来创建矩阵
    ax0.imshow(imgP) #在轴上显示图片，绘制上面的图像矩阵

    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    #下面开始绘制每一簇的数据点
    for i in range(numCluster):
        pointsInCluster = datMat[nonzero(clusterAssing[:, 0].A ==i)[0], :]
        markerStyle = scatterMarkers[i %len(scatterMarkers)]
        #flatten返回矩阵的平整副本。
        ax1.scatter(pointsInCluster[:, 0].flatten().A[0],\
                    pointsInCluster[:, 1].flatten().A[0],\
                    marker=markerStyle, s=90)
    #下面开始绘制质心
    ax1.scatter(myCentroids[:, 0].flatten().A[0],
                myCentroids[:, 1].flatten().A[0],
                marker='+', s=300)
    plt.show()
if __name__ == '__main__':
    # testBiKMeans()
    # clusterClubs(4)
    clusterClubs(6)
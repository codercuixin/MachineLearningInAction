#coding:utf-8
from numpy import  *
import numpy as np;
def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]
def loadExData2():
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]
def loadExData3():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
def getNumberOfSigularValues(Sigma):
    allSum = sum(square(Sigma));
    print 'All sum:', allSum
    curSum = 0.0
    for i in range(len(Sigma)):
        curSum += Sigma[i] **2;
        print curSum/allSum
        if curSum/allSum >= 0.9:
            return i+1;
def euclidSim(inA, inB):
    '''
   基于欧几里得距离相似度测量函数，
    :param inA:都是假设数据是基于列向量进行表示的
    :param inB:
    :return:
    '''
    # print linalg.norm([3, 4, 12]) #默认是3^2+4^2+12^2 开根号，当然还有一些其他的功能
    return 1.0/(1.0+ linalg.norm(inA - inB));
def pearsonSim(inA, inB):
    '''
    利用皮尔森相关系数计算相似度，在lcorrcoef
    :param inA:都是假设数据是基于列向量进行表示的
    :param inB:
    :return:
    '''
    if len(inA)<3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1] #corrcoef[-1, 1]
def cosSim(inA, inB):
    '''
    计算两个向量之间cos夹角的值来计算相似度,并归一化到[0, 1]
    :param inA:都是假设数据是基于列向量进行表示的
    :param inB:
    :return:
    '''
    molecule = float(inA.T * inB); #分子 将1*1矩阵转化成float值
    denominator = linalg.norm(inA) * linalg.norm(inB);#分母
    return 0.5 + 0.5* (molecule / denominator) #cos夹角的值位于[-1, 1],所以需要归一化到[0, 1]
def standEst(dataMat, user, simMeas, item):
    '''
    用来计算再给定相似度计算方法的条件下，用户对物品的估计评分值
    :param dataMat: 数据矩阵，行对应用户，列代表物品
    :param user: 用户编号
    :param simMeas: 相似值计算方法
    :param item: 物品编号
    :return:
    '''
    n = shape(dataMat)[1]
    simiTotal = 0.0; ratSimTotal = 0.0;
    for j in range(n):
        userRating = dataMat[user, j];
        if userRating==0: continue;
        #寻找两个用户都评级过的商品
        overLap = nonzero(logical_and(dataMat[:, item].A>0, dataMat[:, j].A>0))[0]
        #dataMat[:, item].A>0生成True，False组成的列向量, >具有broadcasting的效果，将>后面的0补成前面一样的shape的ndarray或matrix
        #logical_and逻辑与，只有当前后两个同时为True才为True，
        #nonzero最终获得了对应同时为正的下标，这样的数组过滤实际是为了减少计算量
        print 'item: ', item
        print 'overLap: ', overLap
        if len(overLap) ==0: similarity = 0
        #如果可以找到两个用户都评级过的商品，那么就计算商品之间的相似度
        else: similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print 'the %d and %d similarity is: %f'%(item, j, similarity)
        simiTotal += similarity
        ratSimTotal += userRating * similarity
    if simiTotal==0: return 0
    else: return ratSimTotal/simiTotal
xformedItems = [];#保存这个值，避免每次都要计算一次
def svdEst(dataMat, user, simMeas, item):
    '''
    先对数据进行svd降低维度，然后在使用simMeas统计相似度
    :param dataMat:
    :param user:
    :param simMeas:
    :param item:
    :return:
    '''
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0;
    global  xformedItems;
    if xformedItems == []:
        U, Sigma, VT = linalg.svd(dataMat) #奇异值分解 (m, m) ndarray[n], (n, n)
        number = getNumberOfSigularValues(Sigma)
        SigNumber = mat(eye(number) * Sigma[:number]) #构造对角矩阵,(number, number)
        xformedItems = dataMat.T * U[:, :number] * SigNumber.I #(n,m) *(m, number) * (number, number) = (n, number)
        print 'xformedItems', xformedItems;


    for j in range(n):
        userRating = dataMat[user, j]
        if userRating ==0 or j == item: continue; #为0的Item只能与不为0的一起计算相似度
        print xformedItems[item, :].T
        print xformedItems[j, :].T
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T) #因为此时已经是(n, number)，所以找到行向量在转置
        print 'the %d and %d similarity is %f' %(item, j, similarity)
        simTotal += similarity;
        ratSimTotal += similarity * userRating;
    if simTotal == 0: return 0;
    else: return ratSimTotal/simTotal;
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    '''
    产生最高的N个推荐结果
    :param dataMat:
    :param user: 用户编号
    :param N: 要获得几个推荐结果
    :param simMeas: 相似度计算方法
    :param estMethod: 估计方法
    :return:
    '''
    #获取未评分列表，即将等于0的项过滤出来
    unratedItems = nonzero(dataMat[user,:].A ==0)[1]
    if len(unratedItems) ==0 : return 'you has rated everything'
    itemScores = []
    print 'unratedItems', unratedItems
    for item in unratedItems:
        #产生该物品的预计得分
        estimatedScore = estMethod(dataMat, user,simMeas, item)
        # 将item和预计得分都放到itemScores中去
        itemScores.append((item, estimatedScore))
    #从大到小排序，并选择排序后的前N个
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]
def printMat(inMat, thresh=0.8):
    '''打印matrix'''
    for i in range(shape(inMat)[0]):
        for j in range(shape(inMat)[1]):
            if float(inMat[i, j])> thresh:
                print 1,
            else:
                print 0,
        print ''

def imgCompressed(thresh=0.8):
    '''
    基于给定的奇异值值数目来重构图像
    :param numSVD:
    :param thresh:
    :return:
    '''
    myMat = []
    #读取内容
    for line in open('0_5.txt').readlines():
        lineArr = [];
        for i in range(32):
            lineArr.append(int(line[i]));
        myMat.append(lineArr)

    myMat = mat(myMat)
    print '***original matrix***'
    printMat(myMat, thresh)

    U, Sigma, VT = linalg.svd(myMat) #使用svd降维,(m,m) arr[n) (n, n)
    numSVD = getNumberOfSigularValues(Sigma)
    print numSVD
    SigmaNumSVD = mat(zeros((numSVD, numSVD)))
    for i in range(numSVD):
        SigmaNumSVD[i, i] = Sigma[i]
    reconMat = U[:, :numSVD] * SigmaNumSVD * VT[:numSVD, :]#(m, numSVD) *(numSVD,numSVD) *(numSVD, n)
    print '***reconstructed matrix %d singular values***'%(numSVD)
    printMat(reconMat, thresh)




if __name__ == '__main__':
    # data = loadExData();
    # U, Sigma, VT = linalg.svd(data, full_matrices=True);#full_metrics默认为True
    # U:(m, n), Sigma返回的n个元素的数组， VT (n, n)
    # print 'U: ', shape(U)
    # print 'Sigma: ',shape(Sigma)
    # print Sigma
    # print 'VT: ', shape(VT)
    # print Sigma #由于后两个相对于前三个非常小，所以将其舍弃
    # number = getNumberOfSigularValues(Sigma)
    #
    # RetainSegma = mat(diag(Sigma[:number]))
    # print RetainSegma
    #
    # closeData = U[:, :number] * RetainSegma * VT[:number, :]
    # print closeData

    #测试相似度计算函数
    #测试基于欧几里得距离相似度测量函数
    # myMat = mat(loadExData())
    # print euclidSim(myMat[:, 0], myMat[:, 4])#0.1336766024 第0列和第4列确实不怎么相似
    # print euclidSim(myMat[:, 0], myMat[:, 0])#1.0
    #
    # #测试基于皮尔森相关系数的相似度测量函数
    # print pearsonSim(myMat[:, 0], myMat[:, 4])
    # print pearsonSim(myMat[:, 0], myMat[:, 0])
    # #测试基于向量夹角cos值的大小的相似度测量函数
    # print cosSim(myMat[:, 0], myMat[:, 4])
    # print cosSim(myMat[:, 0], myMat[:, 0])

    #测试推荐算法
    # myMat = mat(loadExData2());
    # print myMat
    # print recommend(myMat, user=2)
    # print recommend(myMat, user=2, simMeas=euclidSim)
    # print recommend(myMat, user=2, simMeas=pearsonSim)


    #测试更为真实的矩阵loadadExData3
    # myData3 = mat(loadExData3())
    # U, Sigma, VT = linalg.svd(myData3) #奇异值分解，(m, m) (n,)长度为n的array, (n, n)
    # print recommend(myData3, user=1, estMethod=svdEst)
    # print recommend(myData3, user=1, estMethod=svdEst, simMeas=pearsonSim)
    imgCompressed()
















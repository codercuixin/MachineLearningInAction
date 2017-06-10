#!--**coding:utf-8**--
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
def createCandidate0(dataSet):
    '''
    创建大小为1的所有候选项的集合Candidate0
    :param dataSet:
    :return:
    '''
    Candidate0 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in Candidate0:
                Candidate0.append([item])
    Candidate0.sort()
    #使用frozenset构建一个不可变的无序的独特元素集合。
    #因为下面scanDataSet统计支持度时会以这个集合有键
    #[frozenset([1]), frozenset([2]), frozenset([3]), frozenset([4]), frozenset([5])]
    return map(frozenset, Candidate0)

def scanDataset(DataSet, CandidateK, minSupport):
    '''
    返回候选项集中支持度大于等于最小支持度(出现的次数在总数据的占比）的候选项，以及所有候选项集的支持度
    :param DataSet: 数据集
    :param CandidateK: 候选项集
    :param minSupport: 最小支持度
    :return: retList, supportData
    '''
    supportCount = {} #用来统计支持的个数
    for transaction in DataSet:
        for candidate in CandidateK:
            if candidate.issubset(transaction):
                if not supportCount.has_key(candidate): supportCount[candidate] = 1;
                else: supportCount[candidate] += 1
    dataLength = float(len(DataSet))
    retList = [] #用来统计>minSupport的CandidateK
    supportData = {} #用来统计所有CandidateK的支持度
    for candidate in supportCount:
        support = supportCount[candidate] / dataLength
        if support >= minSupport:
            retList.insert(0, candidate)
        supportData[candidate] = support
    return retList, supportData

def aprioriGen(livingK_1, k):
    '''
    由k-1次中成功活着的候选集中生成第k次的候选集candidatesK
    :param livingK_1: 使用第k-1次大于等于最小支持度的剩余候选集作为新的数据集livingK_1
    :param k: 表示第k次生成候选集
    :return:
    '''
    candidateK = [] #本次候选集
    lenData = len(livingK_1)
    for i in range(lenData):
        for j in range(i+1, lenData):
            dataI = list(livingK_1[i])[:k - 1]
            dataJ = list(livingK_1[j])[:k - 1]
            dataI.sort();dataJ.sort()
            # if(i == 0 and k ==1): print "dataI: ",dataI,"dataJ: ",dataJ
            # 特别的当生成的是包含两个元素的candidate时，上面的dataI，dataJ都是[]
            #如果第I行数据与第J行数据的前k-1个包含的元素相同，那么就将这两行元素的并集加到候选集中
            if dataI == dataJ:
                candidateK.append(livingK_1[i] | livingK_1[j]) #在Python中|表示并操作
    return candidateK
def apriori(dataSet, minSupport=0.5):
    '''
    寻找频繁项集
    :param dataSet:
    :param minSupport:
    :return:
    '''
    candidate0 = createCandidate0(dataSet)
    dataSet = map(set, dataSet)
    living0, supportData0 = scanDataset(dataSet, candidate0, minSupport)

    livings = [living0] #存储所有满足要求的候选集，1个元素的，2个元素的，...
    k = 1;
    while (len(livings[ k -1]) > 0):
        #获得第k的候选集
        candidateK = aprioriGen(livings[k - 1], k)
        #选出满足要求的候选集作为livingK，以及当前支持度
        livingK, supportDataK = scanDataset(dataSet, candidateK, minSupport)
        supportData0.update(supportDataK)
        #将满足要求的候选集加入列表
        if len(livingK) > 0:
            livings.append(livingK)
            k += 1;
        else:
            break;
    return livings, supportData0
def generateRules(livings, supportData, minConf = 0.7):
    '''
    生成关联规则
    :param livings: 频繁项集列表, 例如：[[frozenset([1]), frozenset([3]),
            frozenset([2]), frozenset([5])], [frozenset([1, 3]),
            frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])],
            [frozenset([2, 3, 5])]]
    :param supportData:频繁项集支持度数据的字典
    :param minConf:最小可信度阈值
    :return:
    '''
    bigRuleList = []
    for i in range(1, len(livings)): #只获取两个或更多元素的集合
        for freqSet in livings[i]:
            H1 = [frozenset([item]) for item in freqSet] #H1为频繁项中每一个元素的frozenset集
            if i>1 :  #如果频繁项集的元素个数超过2个，那么会对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #计算规则的可信度，以及找到满足最小可信度要求的规则
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList
def calcConf(freqSet, H, supportData, bigRuleList, minConf =0.7):
    '''
    计算规则的可信度，以及找到满足最小可信度要求的规则
    :param freqSet:
    :param H: 可以出现在规则右边的元素列表，[frozenset([item]) for item in freqSet]
    :param supportData:
    :param bigRuleList:
    :param minConf:
    :return:
    '''
    prunedH = [] #用来保留规则
    for conseq in H:
        #计算可信度
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            #如果满足最小可信度要求，就加入规则列表
            print freqSet-conseq, '-->', conseq, 'conf:',conf;
            bigRuleList.append([freqSet-conseq, conseq, conf])
            prunedH.append(conseq)
    return prunedH
def rulesFromConseq(freqSet, H, supportData, bigRuleList, minConf = 0.7):
    '''
    用于生成候选规则集合
    :param freqSet:频繁项集
    :param H: 可以出现在规则右边的元素列表
    :param supportData:
    :param bigRuleList:
    :param minConf:
    :return:
    '''
    m = len(H[0])
    print "freqSet", freqSet
    print "H", H
    if (len(freqSet) > (m+1)): #检查频繁项集是否达到可以移除大小为m的子集
        #只有len(freqSet)最小为m+2时，才可以生成下一代
        print 'len(freqSet):',len(freqSet)," m+1:", (m+1)
        Hm = aprioriGen(H, m)
        print "Hm", Hm
        # 计算规则的可信度，以及找到满足最小可信度要求的规则
        Hm = calcConf(freqSet, Hm, supportData, bigRuleList, minConf)
        if(len(Hm)> 1): #递归来一次
            rulesFromConseq(freqSet, Hm, supportData, bigRuleList, minConf)
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning



if __name__ == '__main__':
    # dataSet =loadDataSet();
    # print dataSet
    # candidate0 = createCandidate0(dataSet)
    # print candidate0
    #
    # living0, supportData0 = scanDataset(dataSet, candidate0, 0.5)
    # print living0
    # print supportData0

    #测试apriori
    # dataSet =loadDataSet();
    # livings, supportData = apriori(dataSet, 0.5)
    # print livings
    # print livings[0]
    # print livings[1]
    # print livings[2]

    # print aprioriGen(livings[0], 1)
    # print aprioriGen(livings[1], 2)
    # print aprioriGen(livings[2], 3)

    # livings, supportData = apriori(dataSet, 0.5)
    # print livings

    #测试关联规则生成函数generateRules
    # dataSet =loadDataSet();
    # livings, supportData = apriori(dataSet, minSupport=0.5)
    # rules = generateRules(livings, supportData, minConf=0.5)
    # print rules

    #测试getActionIds方法
    # actionIdList, billTitlleList =  getActionIds()
    # print actionIdList
    # print billTitlleList

    #找出包含有毒特征值2的频繁项集
    mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    livings, supportData = apriori(mushDataSet, minSupport=0.3)
    print livings
    for i in range(len(livings)):
        for item in livings[i]:
            if item.__contains__('2'):print item;




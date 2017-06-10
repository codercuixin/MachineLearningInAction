import numpy as np
from os import listdir
import kNN
#change the img into a 1*1024 matrix
def img2Vector(filename):
    returnVector = np.zeros([1,1024])
    file = open(filename)
    for i in  range(32):
        lineStr = file.readline()
        for j in range(32) :
            returnVector[0,i*32+j]= (int)(lineStr[j])
    return  returnVector
def handWritingClassTest():
    hwLables = []
    #get the fileName list in the directory
    trainingFileList = listdir('digits/trainingDigits')
    trainingDatalength = len(trainingFileList)
    #set the traning martrix
    trainingMat = np.zeros((trainingDatalength,1024))
    print trainingDatalength
    for i in range(trainingDatalength):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        # print (int)(fileStr.split('_')[0])
        classNumStr = int(fileStr.split('_')[0])
        print "%d:%d"%(i,classNumStr)
        hwLables.append(classNumStr)

        trainingMat[i,:] = img2Vector("digits/trainingDigits/%s"%fileNameStr)
    print('hwLabels')
    print hwLables
    testFileList = listdir('digits/testDigits')
    testDataLength = len(testFileList)
    errorCount = 0.0
    for i in range(testDataLength):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector("digits/testDigits/%s"%fileNameStr)
        classfiyResult = kNN.classify0(vectorUnderTest , trainingMat , hwLables , 3)
        print "the clasifier came back with : %d ,the real answer is %d" %(classfiyResult,classNumStr)
        if(classNumStr != classfiyResult): errorCount += 1.0
    print 'the total error count is %d\n the error rate is %f' %(errorCount,errorCount/(float)(testDataLength))
# resultVector = img2Vector('digits/testDigits/0_0.txt')
# print resultVector[0,0:31]
# print resultVector[0,64:95]
if __name__ == '__main__':
    handWritingClassTest()

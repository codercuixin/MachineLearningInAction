#coding:utf-8
#获得用户输入
# user = raw_input("Enter your user_name")
# print user
#help(raw_input)
def add(a,b):
    """
    求和
    :param a: 加数
    :param b: 被加数
    :return:
    """
    return a+b
if __name__ == '__main__':
    # print add(3, 2)
    # for item in ["hello","It's me", "I am in california"]:
    #     print item ,
    # print
    # name = "YellowStar5"
    # age = 20
    # print "Your name is %s ,age is %d" %(name, age)
    # pystr = "Python"
    # isCool = ' is cool'
    # print pystr[0]
    # print pystr[-1]
    # print pystr[:2]
    # print pystr[2:]
    # print pystr[1:5]
    # print pystr + isCool
    # print '\t'* 2, pystr*2

    # #列表
    # aList = [1,2,'3', 4.9]
    # print aList
    # print aList[0], aList[-1]
    # print aList[:2]
    # print aList[2:]
    # print aList[0:4]
    #
    # #元组
    # aTuple =  (1,2,'3', 4.9)
    # print aTuple
    # aTuple[0] = 100 #报错，不允许更改

    # aDict = {'host': 'earth'}
    # aDict['port'] = 80
    # aDict[1] = 'Yo'
    # print aDict
    # print aDict.keys()
    # print aDict.values()
    # print aDict.items()

    squared = [x**2 for x in range(4)]
    print squared
    sqdEven = [x**2 for x in range(8) if x%2==0 ]
    print sqdEven
class FooClass(object):
    """my very first class: FooClass"""
    version = 0.1 # class (data) attribute
    def __init__(self, nm='John Doe'):
        """constructor"""
        self.name = nm # class instance (data) attribute
        print 'Created a class instance for', nm
    def showname(self):
        """display instance attribute and class name"""
        print 'Your name is', self.name
        print 'My name is', self.__class__.__name__
    def showver(self):
        """display class(static) attribute"""
        print self.version # references FooClass.version
    def addMe2Me(self, x): # does not use 'self'
        """apply + operation to argument"""
        return x + x



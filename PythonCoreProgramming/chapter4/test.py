#coding:utf-8
def typeCheck(num):
    if isinstance(num, (int, long, float, complex)):
        print 'a number of type: ', type(num).__name__;
    else:
        print num, 'is not a number at all'
    isinstance()
def displayNumType(num):
    #调用两次type函数
    #改成type(num) is types.IntType 就可省下来一次
    if type(num) == type(0):
        print num, ' is a int'
    elif type(num) == type(0.0):
        print num, ' is a float'
    elif type(num) == type(0L):
        print num, ' is a long'
    elif type(num) == type(0 + 0j):
        print num, 'is a complex'
    else:
        print num, 'is not a number'
from types import  IntType, FloatType, LongType, ComplexType
def displayNumType2(num):
    #调用两次type函数
    #改成type(num) is types.IntType 就可省下来一次
    if type(num) is IntType:
        print num, ' is a int'
    elif type(num) is FloatType:
        print num, ' is a float'
    elif type(num) is LongType:
        print num, ' is a long'
    elif type(num) is ComplexType:
        print num, 'is a complex'
    else:
        print num, 'is not a number'
if __name__ == '__main__':
    # print bool(1)
    # print bool(True)
    # print bool(0)
    # print bool('1')
    # print bool('0')
    # print bool([])
    # print bool((1,))
    #
    # foo = 42
    # bar = foo <100
    # print bar
    # print bar + 100

    # #无__nonzero()
    # class C:pass
    # c = C()
    # print bool(c)
    # print bool(C)
    #
    # class C:
    #     def __nonzero__(self):
    #         return False;
    #
    # c = C()
    # print bool(c)
    # print bool(C)
    # #永远不要这么干
    # True, False = False, True
    # print bool(False), bool(True)
    import  random
    print random.randrange(1, 10, 2);
    print range(1, 10, 2)
    print random.random();
    print random.uniform(0, 1)
    print random.choice(seq=[1, 2, 3])













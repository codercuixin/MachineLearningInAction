from numpy import *;
randMat =mat(random.rand(4,4));
invRandMat = randMat.I
myeye= randMat*invRandMat
print randMat
print invRandMat
print myeye
print myeye-eye(4)
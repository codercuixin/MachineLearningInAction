import  numpy as np
x = np.array([[1,2,3] , [4,5,6]])
# print type(x) #(type 'numpy.ndarray')
# print x.shape #(2L,3L)
# print x.shape[0] #2
# print x.shape[1] #3
# print x.dtype  #int32

#can be indexed ,index start from 0,
# print x[0,1] #2
# print x[1,2] #6


#For example slicing can produce views of the array:
#select a cloumn into an array ,or you can select a row into an array
# z = x[0,:] #array [1,2,3]
# print z
# y = x[:,1]
# print y #array([2, 5])

# y[0] = 9 # the data it is referring to is taken care of by the base ndarray.
#         # this also changes the corresponding element in x
# print y #array([9, 5])
# print x #array([[1, 9, 3],[4, 5, 6]])
w = x[0:1,:] # you can select [0,1)  row into a new matrix [[1 2 3][4 5 6]]
print w
u = x [:, 0:1] # you can select [0,1) coloum into a new matrix [[1][4]]
print u

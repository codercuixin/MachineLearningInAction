#!--**coding:utf-8--**
from numpy import *
from Tkinter import *
import regTrees
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get(): #如果选中Model Tree的话,就使用模型树相关的方法来创建和预测
        if tolN < 2: tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForecast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForecast(myTree, reDraw.testDat)
    reDraw.a.scatter(array(reDraw.rawDat[:,0]), array(reDraw.rawDat[:,1]), s=5) #画出散点图
    reDraw.a.plot(reDraw.testDat, yHat, linewidth =2.0) #plot方法构建连续曲线
    reDraw.canvas.show()
def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

def getInputs():
    #验证一下数据类型，如果不符合就返回默认值
    try: tolN = int(tolNEntry.get())
    except:
        tolN = 10
        print "enter Integer for tolN"
        tolNEntry.delete(0, END)
        tolNEntry.insert(0, '10')

    try: tolS = float(tolSEntry.get())
    except:
        tolS = 1.0
        tolSEntry.delete(0, END)
        tolSEntry.insert(0, '1.0')
    return tolN, tolS

root = Tk()
reDraw.f = Figure(figsize=(5, 4), dpi =100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

# #第0行的说明
# Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)

#第一行的tolN
Label(root, text='tolN').grid(row=1, column=0)
tolNEntry = Entry(root)
tolNEntry.grid(row=1, column=1)
tolNEntry.insert(0,'10')
#第二行的tolS
Label(root, text="tolS").grid(row=2, column=0)
tolSEntry = Entry(root)
tolSEntry.grid(row=2, column=1)
tolSEntry.insert(0, '1.0')
#横跨row[1,4]的ReDraw按钮
Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2,rowspan=3)

Button(root, text='Quit', command=root.quit).grid(row=1, column=2)
#复选框，是否构建模型树
chkBtnVar = IntVar() #整数值，用来读取复选框的状态
chkBtn = Checkbutton(root, text='Model Tree', variable= chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

# reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
# reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0], 0.01))
# #原数据第一列的最小值最为开始，最大值作为结束，步长为0.01
# reDraw(1.0, 10)
reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
print reDraw.rawDat
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0, 10)
root.mainloop()
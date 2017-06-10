#coding:utf-8
import os
if __name__ == '__main__':
    #获得文件名
    while True:
        fname = raw_input('Enter file name: ');
        if os.path.exists(fname):
            print 'file: %s has already existed' %fname
        else:
            break;
    fileContent = []
    print '\n Enter file content， if you want quit, type .'
    while True:
        entry = raw_input('>')
        if entry == '.':
            break;
        else:
            fileContent.append(entry)
    #将从fileContent写入到文件中去
    fw = open(fname, 'w') #以写模式打开
    fw.write('\n'.join(fileContent))
    fw.flush()
    fw.close()
    print 'Done!'

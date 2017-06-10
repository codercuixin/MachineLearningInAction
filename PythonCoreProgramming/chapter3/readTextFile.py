
if __name__ == '__main__':
    fileName = raw_input('Enter file name: ')
    print
    try:
        fr = open(fileName, 'r')
    except IOError, e:
        print 'file open error', e
    else:
        for line in fr:
            print line;
        fr.close()

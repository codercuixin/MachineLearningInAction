import sys
import numpy as np
def read_input(file):
    for line in file:
        yield line.rstrip();

input = read_input(sys.stdin);
input = [float(line) for line in input];
numInputs = len(input)
input = np.mat(input)
squareInput = np.power(input, 2)
print '%d\t%f\t%f'%(numInputs, np.mean(input), np.mean(squareInput))
print >> sys.stderr, 'report: still alive'

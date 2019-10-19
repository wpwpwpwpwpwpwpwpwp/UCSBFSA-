from search import bfs
from search import ucs
from search import astar_path
import re
import os

data = []
file = r'testcases/input1.txt'
for line in open(file, "r"):
    data.append(line.replace("\n", ""))
print(data)
##################First line##################
if data[0] == 'BFS':
    search = bfs
elif data[0] == 'UCS':
    search = ucs
else:
    search = astar_path
######Second line#######
grid = data[1].split(" ")
W = grid[0]
H = grid[1]
#######Third line##########
startPoint = data[2].split(" ")
startPoint = [int(startPoint[0]), int(startPoint[1])]
#######Fourth line###########
heightDifference = int(data[3])
########Fifth line##########
number = int(data[4])
##########Next h lines##########
matrix = []
for i in range(5 + number, 5 + number + int(H)):
    matrix.append(re.split(" ", data[i]))
print(matrix)
m = []
for a in range(int(W)):
    mm = []
    for b in range(int(H)):
        mm.append(0)
    m.append(mm)

for a in range(int(W)):
    for b in range(int(H)):
        m[a][b] = int(matrix[b][a])
print(m)
output = r'F:\1019\output\output1.txt'
"""
try:
    os.remove(output)
    print(os.remove(output))
except:
    print('nothing to clean')
"""
startPoint1 = '{}-{}'.format(startPoint[0], startPoint[1])
print(startPoint1)
##########Next N lines##########
for i in range(5, 5 + number):
    endPoint1 = data[i].split(" ")
    endPoint = [int(endPoint1[0]), int(endPoint1[1])]
    endPoint = '{}-{}'.format(endPoint[0], endPoint[1])
    print(endPoint)
    k = search(m, startPoint1, endPoint, heightDifference)
    print(k)
    f = open(output, 'a')
    if k == 'FAIL':
        f.write(k)
    else:
        for ii in range(len(k)):
            if (ii == (len(k)) - 1):
                f.write(k[ii].replace("-", ","))
            else:
                f.write(k[ii].replace("-", ",") + " ")
    if i == number + 4:
        print('EOF')
    else:
        f.write('\n')
    f.close()
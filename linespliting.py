import cv2
import numpy as np

img = cv2.imread("post_skew.png",0)
output = img
binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite("resultbinary.jpg", binary)
coords = [[ 1316.9988147540982, 2510.9986295081962, 75.0002168852459, 623.9996250819672 ],
[ 23.99999218032787, 1236.0013348032787, 98.99982295081968, 1390.0002442622952 ],
[ 1007.0001491803279, 1203.000149672131, 145.99985590163934, 225.99982983606557 ],
[ 889.9987983606557, 1209.9986940983608, 257.0001254098361, 339.0001737704918 ],
[ 1322.0011459016393, 2548.0012934426227, 583.9985655737705, 3061.997254098361 ],
[ 30.000000950819672, 1241.0017353770493, 1423.0001852459018, 1800.0000570491804 ],
[ 36.00000972131147, 1243.9986293934428, 1792.001475409836, 2259.0029803278685 ],
[ 1116.0003442622951, 1244.0003883606555, 2250.0022163934427, 2327.0020786885248 ],
[ 490.0010737704918, 696.0010173770493, 2254.000649180328, 2340.0008463934428 ],
[ 351.0001913114754, 478.00019819672127, 2268.999062295082, 2416.9989926229505 ],
[ 42.99988409836065, 1247.0000709836065, 2340.001275409836, 2756.0013114754097 ],
[ 44.99995852459016, 1251.99897, 2770.9997245901636, 3073.9998457377046 ]]



character = []
Numberofitems = len(coords)
rowsplit = []
for i in range(Numberofitems):
    rowsplit.append([])
    coords[i] = list(map(int, coords[i]))
threshold = 245
for s in range(Numberofitems):
    rowflag = 1
    for i in range(coords[s][2], coords[s][3]):
        sum = 0
        count = coords[s][1] - coords[s][0]
        for j in range(coords[s][0], coords[s][1]):
            sum = sum + binary[i][j]
        print(sum / count)
        if sum / count > threshold and rowflag == 0:
            continue
        elif sum / count > threshold and rowflag == 1:  # just get out of the block
            t1 = i
            #rowsplit[s].append(i)
            rowflag = 0
            flag = 0
            #cv2.line(output, (coords[s][0], i), (coords[s][1], i), (0, 0, 0), 1)
            continue
        elif sum / count <= threshold and rowflag == 0:  # just get into the block
            #rowsplit[s].append(i)
            if flag==0:
                t2 = i
            rowsplit[s].append((t1+t2)/2)
            rowflag = 1
            cv2.line(output, (coords[s][0], (t1+t2)/2), (coords[s][1], (t1+t2)/2), (0, 0, 0), 1)
            continue
        else:
            continue
for i in range(Numberofitems):
    cv2.rectangle(output, (coords[i][0], coords[i][2]),
                  (coords[i][1], coords[i][3]), 3)

'''
for s in range(Numberofitems):
    splitnum=len(rowsplit[s])
    for k in range(splitnum-1):
        colflag = 1
        tempcol=[]
        for i in range(coords[s][0], coords[s][1]):
            summ = 0
            countt = rowsplit[s][k+1] - rowsplit[s][k]
            for j in range(rowsplit[s][k], rowsplit[s][k+1]):
                summ = summ + binary[j][i]
            # print(sum / count)
            if summ / countt > threshold and colflag == 0:
                continue
            elif summ / countt > threshold and colflag == 1:  # just get out of the block
                #rowsplit[s].append(i)
                colflag = 0
                cv2.line(output, (i,rowsplit[s][k]), (i,rowsplit[s][k+1]), (0, 0, 0), 1)
                continue
            elif summ / countt <= threshold and colflag == 0:  # just get into the block
                #rowsplit[s].append(i)
                colflag = 1
                cv2.line(output, (i,rowsplit[s][k]), (i,rowsplit[s][k+1]), (0, 0, 0), 1)
                continue
            else:
                continue'''
#return rowsplit
cv2.imwrite("resultpost_skew.jpg", output)


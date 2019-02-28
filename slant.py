#https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
import cv2
import numpy as np
import sys
sys.setrecursionlimit(100000)

def Flood(image, index, label, x, y, i):
    #The coordinate of first pixel is (x,y); The index of component is i
    (height, width)=image.shape[:2]
    list=[[x,y]]
    while (not list) == False:
        cx, cy = list.pop()
        index[cx, cy] = i
        label[cx, cy] = 1
        pixel = image[cx, cy]
        if cx - 1 >= 0 and index[cx - 1, cy] == 0 and image[cx - 1, cy] == pixel:
            list.append([cx - 1, cy])
            #Flood(image, index, x - 1, y, i, c, height, width)
        if cx + 1 < height and index[cx + 1, cy] == 0 and image[cx + 1, cy] == pixel:
            list.append([cx + 1, cy])
            #Flood(image, index, x + 1, y, i, c, height, width)
        if cy - 1 >= 0 and index[cx, cy - 1] == 0 and image[cx, cy - 1] == pixel:
            list.append([cx, cy - 1])
            #Flood(image, index, x, y - 1, i, c, height, width)
        if cy + 1 < width and index[cx, cy + 1] == 0 and image[cx, cy + 1] == pixel:
            list.append([cx, cy + 1])
            #Flood(image, index, x, y + 1, i, c, height, width)
    return index, label

def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)

    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])"""
    max = np.max(pts, axis=0)
    min = np.min(pts, axis=0)
    topleft = np.array([min[0], min[1]])
    #topright = np.array([min[0], max[1]])
    #bottomleft = np.array([max[0], min[1]])
    bottomright = np.array([max[0], max[1]])
    return np.array([topleft,bottomright])

if __name__ == '__main__':
    # Reading the input image
    img = cv2.imread('post_skew.png', 0)
    (h, w) = img.shape[:2]
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(img)

    # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_dilation = cv2.dilate(thresh, kernel, iterations=10)
    blur = cv2.GaussianBlur(img_dilation, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imwrite("binary_resultprojectivetrans.jpg", thresh)

    #REQUIRE_GROUPING=img_dilation
    LABELED_OR_NOT=np.zeros([h,w])
    PIXEL_GROUP=np.zeros([h,w])


    #sum=0
    for i in range(h):
        for j in range(w):
            if thresh[i][j]==0:
                LABELED_OR_NOT[i][j] = 1
                #sum=sum+1


    num=0 #The index of component
    #while True:
        #r1 = np.int(np.random.random() * h) - 1
        #r2 = np.int(np.random.random() * w) - 1
    for r1 in range(h):
        for r2 in range(w):
            if LABELED_OR_NOT[r1][r2] == 0:
                print(r1,r2)
                num = num + 1
                print(num)
                PIXEL_GROUP, LABELED_OR_NOT=Flood(thresh, PIXEL_GROUP, LABELED_OR_NOT, r1, r2, num)


    '''
    cv2.namedWindow('Dilation', 0)
    cv2.imshow("Dilation", img_dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    list=[]
    point=[]
    for i in range(num):
        list.append([])
        point.append([])

    for i in range(h):
        for j in range(w):
            if PIXEL_GROUP[i][j] != 0:
                list[int(PIXEL_GROUP[i][j])-1].append([i,j])

    print(len(list))
    for i in range(num):
        point[i]=fourCornersSort(np.array(list[i]))


    cv2.imwrite("resultblock.jpg", img)


    binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rowsplit = []
    for i in range(num):
        rowsplit.append([])
    threshold = 240
    for s in range(num):
        rowflag = 1
        for i in range(point[s][0][0], point[s][1][0]):
            sum = 0
            count = point[s][1][1] - point[s][0][1]
            for j in range(point[s][0][1], point[s][1][1]):
                sum = sum + binary[i][j]
            print(sum / count)
            if sum / count > threshold and rowflag == 0:
                continue
            elif sum / count > threshold and rowflag == 1:  # just get out of the block
                rowsplit[s].append(i)
                rowflag = 0
                cv2.line(img, (point[s][0][1], i), (point[s][1][1], i), (0, 0, 0), 1)
                continue
            elif sum / count <= threshold and rowflag == 0:  # just get into the block
                rowsplit[s].append(i)
                rowflag = 1
                cv2.line(img, (point[s][0][1], i), (point[s][1][1], i), (0, 0, 0), 1)
                continue
            else:
                continue
    for i in range(num):
        cv2.rectangle(img, (tuple(point[i][0])[1], tuple(point[i][0])[0]),
                      (tuple(point[i][1])[1], tuple(point[i][1])[0]), 3)

    cv2.imwrite("resultsplitpaper.jpg", img)
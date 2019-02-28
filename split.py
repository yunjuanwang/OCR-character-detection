#https://bretahajek.com/2017/01/scanning-documents-photos-opencv/
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load image and convert it from BGR to RGB
image = cv2.cvtColor(cv2.imread("projectivetrans.jpg"), cv2.COLOR_BGR2RGB)
'''cv2.namedWindow('img',0)
cv2.imshow("img",image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
def resize(img, height=800):
    """ Resize image to given height """
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))

# Resize and convert to grayscale
img = cv2.cvtColor(resize(image), cv2.COLOR_BGR2GRAY)

# Bilateral filter preserve edges
# 9:The diameter range of each pixel's neighborhood during filtering
# 75:Sigma values for color space filters
# 75:Sigma values of the filter in coordinate space
img = cv2.bilateralFilter(img, 9, 75, 75)

# Create black and white image based on adaptive threshold
# The function transforms a grayscale image to a binary image
# def adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None):
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

# Median filter clears small details
# def medianBlur(src, ksize, dst=None)
img = cv2.medianBlur(img, 11)

# Add black border in case that page is touching an image border
img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
edges = cv2.Canny(img, 200, 250)


# Getting contours
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Finding contour of biggest rectangle
# Otherwise return corners of original image
# Don't forget on our 5px border!
height = edges.shape[0]
width = edges.shape[1]
MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

# Page fill at least half of image, then saving max area found
maxAreaFound = MAX_COUNTOUR_AREA * 0.5

# Saving page contour
pageContour = np.array([[5, 5], [5, height-5], [width-5, height-5], [width-5, 5]])

# Go through all contours
for cnt in contours:
    # Simplify contour
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

    # Page has 4 corners and it is convex
    # Page area must be bigger than maxAreaFound
    if (len(approx) == 4 and
            cv2.isContourConvex(approx) and
            maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):

        maxAreaFound = cv2.contourArea(approx)
        pageContour = approx

# Result in pageConoutr (numpy array of 4 points):

def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)

    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset

    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt


# Sort and offset corners
pageContour = fourCornersSort(pageContour[:, 0])
pageContour = contourOffset(pageContour, (-5, -5))

# Recalculate to original scale - start Points
sPoints = pageContour.dot(image.shape[0] / 800)

# Using Euclidean distance
# Calculate maximum height (maximal length of vertical edges) and width
height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
             np.linalg.norm(sPoints[2] - sPoints[3]))
width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
            np.linalg.norm(sPoints[3] - sPoints[0]))

# Create target points
tPoints = np.array([[0, 0],
                    [0, height],
                    [width, height],
                    [width, 0]], np.float32)

# getPerspectiveTransform() needs float32
if sPoints.dtype != np.float32:
    sPoints = sPoints.astype(np.float32)

# Wraping perspective
M = cv2.getPerspectiveTransform(sPoints, tPoints)
newImage = cv2.warpPerspective(image, M, (int(width), int(height)))

# Saving the result. Yay! (don't forget to convert colors bact to BGR)
cv2.imwrite("resultincludeeq.jpg", cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))

imgsplitgrey = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
imgsplit = cv2.bilateralFilter(imgsplitgrey, 9, 75, 75)
imgsplit= cv2.adaptiveThreshold(imgsplit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
'''cv2.namedWindow('img',0)
cv2.imshow("img",imgsplit)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

imgsplitheight, imgsplitwidth=newImage.shape[:2]
print(imgsplitwidth)
print(imgsplitheight)
threshold=240
#iterate the column from left to right
#colflag=0 means outside one block, colflag=1 means inside one block
colflag=1
colsplit=[]

for i in range(imgsplitwidth):
    sum=0
    for j in range(imgsplitheight):
        sum=sum+imgsplit[j,i]
    #print(sum/imgsplitheight)
    if sum/imgsplitheight>threshold and colflag==0:
        continue
    elif sum/imgsplitheight>threshold and colflag==1:#just get out of the block
        colsplit.append(i)
        colflag = 0
        cv2.line(imgsplit, (i,0), (i,imgsplitheight), (0,0,0), 1)
        continue
    elif sum/imgsplitheight<=threshold and colflag==0:#just get into the block
        colsplit.append(i)
        colflag=1
        cv2.line(imgsplit, (i,0), (i,imgsplitheight), (0,0,0), 1)
        continue
    else:
        continue

##print(colsplit)
'''cv2.namedWindow('img',0)
cv2.imshow("img",imgsplit)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
cv2.imwrite("splitcolincludeeq.jpg",imgsplit)

'''
i=0
for i in range(imgsplitheight):
    sum=0
    j=0
    for j in range(imgsplitwidth):
        sum=sum+imgsplit[i,j]
    #print(sum/imgsplitheight)
    if sum/imgsplitwidth>threshold and rowflag==0:
        continue
    elif sum/imgsplitwidth>threshold and rowflag==1:#just get out of the block
        rowsplit.append(i)
        rowflag = 0
        cv2.line(imgsplit, (0,i), (imgsplitheight,i), (0,0,0), 1)
        continue
    elif sum/imgsplitwidth<=threshold and rowflag==0:#just get into the block
        rowsplit.append(i)
        rowflag=1
        cv2.line(imgsplit, (0,i), (imgsplitheight,i), (0,0,0), 1)
        continue
    else:
        continue

'''
rowflag=1
rowsplit=[]
for s in range(len(colsplit)-1):
    for i in range(imgsplitheight):
        sum=0
        count=colsplit[s+1]-colsplit[s]
        for j in range(colsplit[s],colsplit[s+1]):
            sum=sum+imgsplit[i,j]
        if sum/count>threshold and rowflag==0:
            continue
        elif sum/count>threshold and rowflag==1:#just get out of the block
            rowsplit.append(i)
            rowflag = 0
            cv2.line(imgsplit, (colsplit[s],i), (colsplit[s+1],i), (0,0,0), 1)
            continue
        elif sum/count<=threshold and rowflag==0:#just get into the block
            rowsplit.append(i)
            rowflag=1
            cv2.line(imgsplit, (colsplit[s],i), (colsplit[s+1],i), (0,0,0), 1)
            continue
        else:
            continue
cv2.imwrite("splitallincludeeq.jpg",imgsplit)

cv2.namedWindow('img',0)
cv2.imshow("img",imgsplit)
cv2.waitKey(0)
cv2.destroyAllWindows()
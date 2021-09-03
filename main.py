import cv2
import sys
import numpy as np
from PIL import Image
import numpy as np
import imutils as imutils
import matplotlib.pyplot as plt
import PIL
from skimage.filters import threshold_local


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped



img = cv2.imread('/Users/somang/Desktop/ex3.png', cv2.IMREAD_GRAYSCALE)
ratio = img.shape[0] / 600.0
orig = img.copy()
img = imutils.resize(img,height = 600)
cv2.imshow("1",img)
gray = cv2.cvtColor(img, cv2.THRESH_OTSU)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray, 75,200)

#cv2.imshow("image",img)
#cv2.imshow("edged",edged)

cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key = cv2.contourArea,reverse = 1) [:5]

for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02 * peri,True)

    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(img,[screenCnt],-1,(0,255,0),2)
#cv2.imshow("Outline",img)


warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)
#T = threshold_local(warped,11,offset = 10, method = "gaussian")
#warped = (warped>T).astype("uint8") * 255
cv2.imshow("Scanned",imutils.resize(warped, height = 650))
scanned = imutils.resize(warped, height = 650)

#흰,검 추출
blur = cv2.GaussianBlur(scanned, (5, 5), 0)
ret, thr2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
scanned = thr2
thr2 = scanned
#cv2.imshow("bw2",thr2)

img_modify = np.zeros(scanned.shape, scanned.dtype)
""""""
for h in range(thr2.shape[0]):
    for w in range(thr2.shape[1]):
        if h>62 and h<75 and w>33 and w<46:
            img_modify[h,w] = 255
            #w = 16, 간격 144
        elif h > 62 and h < 75 and w > 145 and w < 158:
            img_modify[h, w] = 255
        elif h > 62 and h < 75 and w > 260 and w < 271:
            img_modify[h, w] = 255
        elif h > 62 and h < 75 and w > 378 and w < 391:
            img_modify[h, w] = 255
        else:
            img_modify[h,w] = thr2[h,w]

cv2.imshow("0", img_modify)

cv2.waitKey()
cv2.destroyAllWindows()
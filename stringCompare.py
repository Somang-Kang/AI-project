import cv2
import numpy as np
import imutils as imutils


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



img = cv2.imread('/Users/somang/Desktop/ex15.png')
ratio = img.shape[0] / 600.0
orig = img.copy()
img = imutils.resize(img,height = 600)
cv2.imshow("1",img)
#gray = cv2.cvtColor(img, cv2.THRESH_OTSU)
#gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(img, 75,200)

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

#cv2.drawContours(img,[screenCnt],-1,(0,255,0),2)
#cv2.imshow("Outline",img)


warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)
#T = threshold_local(warped,11,offset = 10, method = "gaussian")
#warped = (warped>T).astype("uint8") * 255
cv2.imshow("Scanned",imutils.resize(warped, height = 650))
scanned = imutils.resize(warped, height = 650)
cv2.imshow("scanned",scanned)


height, width = scanned.shape[:2] # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]
img_hsv = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환
lower_blue = (120-10, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_blue = (120+10, 255, 255)
img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) # 범위내의 픽셀들은 흰색, 나머지 검은색

# 바이너리 이미지를 마스크로 사용하여 원본이미지에서 범위값에 해당하는 영상부분을 획득
img_result = cv2.bitwise_and(scanned, scanned, mask=img_mask)

cv2.imshow('img_origin', scanned)
cv2.imshow('img_mask', img_mask)
cv2.imshow('img_color', img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()

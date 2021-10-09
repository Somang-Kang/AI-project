# inRange 함수를 이용해서 특정 색상 검출

import cv2

src = cv2.imread('resize/right_tiger2.png')
# size 축소
src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
cv2.imshow('srchsv', src_hsv)

#  0 < B < 100 ,   128 < G < 255 , 0 < R < 100
dst1 = cv2.inRange(src, (20, 100, 100), (200, 255, 255))
img_result = cv2.bitwise_and(src_hsv, src_hsv, mask=dst1)
#dst2 = cv2.inRange(src_hsv, (100, 100, 0), (200, 255, 255))
dst2 = cv2.inRange(src_hsv, (100, 50, 0), (200, 255, 255))
cv2.imshow('dst2', dst2)
cv2.imwrite("/Users/somang/Desktop/blackandwhiteTiger2.png",dst2)

img_result2 = cv2.bitwise_and(src_hsv, src_hsv, mask=dst2)
#cv2.imshow('dst1', dst1)


#cv2.imwrite("/Users/somang/Desktop/redversion2.png",dst1)

cv2.imshow('src', src)
cv2.moveWindow('src', 400, 100)

cv2.imshow('dst1', dst1)
cv2.moveWindow('dst1', 400, 450)

cv2.imshow('img_result', img_result)
cv2.moveWindow('img_result', 800, 450)

#cv2.imwrite("/Users/somang/Desktop/compare1.png",dst1)

cv2.waitKey()
cv2.destroyAllWindows()


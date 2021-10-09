import cv2
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
import numpy as np


def lineDetection(lines, img_new):
    img = img_new.copy()
    if lines is None:
        lines = []

    for j in lines:
        x1 = (j[0][0] - 15) // 68
        y1 = (j[0][1] - (49 if x1 % 2 == 0 else 14)) // 69
        x1_p = x1 * 68 + 49
        y1_p = y1 * 69 + (83 if x1 % 2 == 0 else 48)

        x2 = (j[1][0] - 15) // 68
        y2 = (j[1][1] - (49 if x2 % 2 == 0 else 14)) // 69
        x2_p = x2 * 68 + 49
        y2_p = y2 * 69 + (83 if x2 % 2 == 0 else 48)

        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            continue

        sz = 25
        if abs(x2_p - j[1][0]) > sz or abs(y2_p - j[1][1]) > sz:
            continue

        if abs(x1_p - j[0][0]) > sz or abs(y1_p - j[0][1]) > sz:
            continue

        # if (x1_p-x2_p)*(j[0][2]- j[0][3]) == (j[0][0]- j[0][1])*(y1_p-y2_p)
        if abs((np.rad2deg(np.arctan2(j[1][1] - j[0][1], j[1][0] - j[0][0])) * -1) % 180 - (
                np.rad2deg(np.arctan2(y2_p - y1_p, x2_p - x1_p)) * -1) % 180) > 10:
            continue

        if np.sqrt((y2_p - y1_p) ** 2 + (x2_p - x1_p) ** 2) * 0.3 > np.sqrt(j[1][1] - j[0][1]) ** 2 + (
                j[1][0] - j[0][0]) ** 2:
            continue

        if x1 == x2 and y1 == y2:
            continue

        cv2.line(img, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 255), 2)

    return img


img = cv2.imread('1004/pig/color_pig_2.jpg')
img2 = img.copy()
# 그레이 스케일로 변환 및 엣지 검출 ---①
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(imgray, 500, 250 )
edges = canny(imgray, 2, 1, 25)

lines = probabilistic_hough_line(edges)
img_new = lineDetection(lines, img)

for j in lines:
    cv2.line(img2, (j[0][0], j[0][1]), (j[1][0], j[1][1]), (0, 255, 255), 2)

cv2.imshow("img1", img_new)
cv2.imshow("img2", img2)
cv2.waitKey()

import cv2
import numpy as np

img = cv2.imread('resize/resize2.png')


cnt = 0
for i in range(15):
	for t in range(10 if i % 2 == 0 else 11):
		# 1048 * 787 이미지 기준 좌표
		x = 68 * i + 49
		y = 69 * t + (83 if i % 2 == 0 else 48)
		area = (int(x - 20), int(y - 20), int(x + 20), int(y + 20))
		crop_img = img[y - 20: y + 20, x - 20: x + 20]

		picname = "/Users/somang/Desktop/croped_img/" + str(cnt) + ".png"

		cnt += 1

		cv2.imwrite(picname, crop_img)







criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 0
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 280     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 500   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.85

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.4

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

#folder = "cow"
#number = "2"

#img = cv2.imread('/Users/somang/Desktop/croped_img/' + number + '.png')
#print("loading pics successfully")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the circle board centers
keypoints = blobDetector.detect(gray) # Detect blobs.

# Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,9), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

if ret == True:
	corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (5,5), (-1,-1), criteria)    # Refines the corner locations.

	im_with_keypoints = cv2.drawChessboardCorners(img, (4,9), corners2, ret)

cv2.imshow("img", im_with_keypoints) # display
cv2.waitKey()

picname = "/Users/somang/Desktop/croped_img/blob_detection/blob_detect.png"

cv2.imwrite(picname,im_with_keypoints)

cv2.destroyAllWindows()

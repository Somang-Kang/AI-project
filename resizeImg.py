import cv2
from PIL import Image
# 자르기

img = Image.open('resize/flat_img.png')

img_resize = img.resize((1048, 787))

img_resize.save('resize/resize2.png')

cv2.destroyAllWindows()

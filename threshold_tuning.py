import cv2
import numpy as np
from main import thresh_pipeline

def nothing(x):
    pass

# Read a testing image.
img =cv2.imread('./test_images/test5.jpg')
cv2.namedWindow('image', flags=0)

# create trackbars for color change
cv2.createTrackbar('s_min', 'image', 70, 255, nothing)
cv2.createTrackbar('s_max', 'image', 250, 255, nothing)
cv2.createTrackbar('sobx_min', 'image', 20, 180, nothing)
cv2.createTrackbar('sobx_max', 'image', 100, 180, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

img2 = np.copy(img)
while(1):
    cv2.imshow('image',img2)
    k = cv2.waitKey(1) & 0xFF   # Press "Esc" to close this function
    if k == 27:
        break


    # get current positions of five trackbars
    s_min = cv2.getTrackbarPos('s_min', 'image')
    s_max = cv2.getTrackbarPos('s_max', 'image')
    sx_min = cv2.getTrackbarPos('sobx_min', 'image')
    sx_max = cv2.getTrackbarPos('sobx_max', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 1:
        img_br = cv2.GaussianBlur(img, (7,7), 0)
    else:
        img_br = np.copy(img)

    img_th = thresh_pipeline(img_br, (s_min, s_max), (sx_min, sx_max)) # note this is one channle with scale 0-1.
    img2 = np.dstack((img_th*255, img_th*255, img_th*255))


cv2.destroyAllWindows()

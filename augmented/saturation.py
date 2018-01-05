import numpy as np
import cv2

# Read the name of all available images into `lines`
liquid_path  = "data/Liquid/"
empty_path   = "data/Empty/"
unknown_path = "data/Unknown/"
text_file = open(liquid_path+'filenames.txt', 'r')
lines = text_file.read().splitlines()

img_name = lines[244]
img = cv2.imread(liquid_path+img_name)

width = 400
height = 400
center_x = int(width / 2)
img = cv2.resize(img, (width, height))
img_original = img
cropped = img[0:height, center_x:center_x+1]

# STEPS TO CHANGE SATURATION
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
(h,s,v) = cv2.split(img)
s = s*1
s = np.clip(s,0,255)
img = cv2.merge([h,s,v])
img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

bgr = [155, 155, 155]
thresh = 70

minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

maskBGR = cv2.inRange(img,minBGR,maxBGR)
resultBGR = cv2.bitwise_and(img, img, mask = maskBGR)

#convert 1D array to 3D, then convert it to HSV and take the first element
# this will be same as shown in the above figure [65, 229, 158]
hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]

minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

maskHSV = cv2.inRange(imgHSV, minHSV, maxHSV)
resultHSV = cv2.bitwise_and(imgHSV, imgHSV, mask = maskHSV)

cv2.imshow("Result BGR", resultBGR)
cv2.imshow("Result HSV", resultHSV)
cv2.imshow("Pre Saturation", img_original)
cv2.imshow("Original IMG", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

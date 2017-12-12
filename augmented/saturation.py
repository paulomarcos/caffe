import numpy as np
import cv2

# Read the name of all available images into `lines`
liquid_path  = "data/Liquid/"
empty_path   = "data/Empty/"
unknown_path = "data/Unknown/"
text_file = open(liquid_path+'filenames.txt', 'r')
lines = text_file.read().splitlines()

img_name = lines[1]
img = cv2.imread(liquid_path+img_name)

img = cv2.resize(img, (400, 400))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
(h, s, v) = cv2.split(hsv)

#s = s*1
#s = np.clip(s, 0, 255)

merge = cv2.merge([h,v,s])

cv2.imshow("hue", h)
cv2.imshow("saturation", s)
cv2.imshow("value", v)
cv2.imshow("normal", img)
cv2.imshow("merge", merge)
cv2.waitKey(0)
cv2.destroyAllWindows()

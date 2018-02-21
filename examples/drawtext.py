import numpy as np
import cv2
import os, errno

img = cv2.imread("/home/gother/caffe-cupstate/data/ilsvrc12/val//n01111111_0001.JPEG")

directory = "classification_results"
try:
    os.makedirs(directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

green = (10, 90, 10)
red = (10, 10, 235)

color = green
font_text = "Liquid"

cv2.rectangle(img,(9,11),(96,39),(155,155,155),-1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,font_text,(16,30), font, 0.55,color,1,cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

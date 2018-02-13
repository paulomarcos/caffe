import numpy as np
import matplotlib.pyplot as plt
import cv2

liquid_path  = "data/Liquid/"
empty_path   = "data/Empty/"
unknown_path = "data/Unknown/"

paths = [liquid_path, empty_path, unknown_path]
width = height = 400

hog = cv2.HOGDescriptor()

for path in paths:
    text_file = open(path+'filenames.txt', 'r')
    lines = text_file.read().splitlines()

    for name in lines:
        img = cv2.imread(path+name)
        img = cv2.resize(img, (width, height))
        hog_img = hog.compute(img)
        print hog_img
        cv2.waitKey(0)

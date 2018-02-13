import numpy as np
import matplotlib.pyplot as plt
import cv2

liquid_path  = "dataIMN/Liquid/"
empty_path   = "dataIMN/Empty/"
unknown_path = "dataIMN/Unknown/"

paths = [liquid_path, empty_path, unknown_path]
liquid = []
empty = []
unknown = []

width = height = 400
center_x = int(width / 2)

s_avg = 0

results = []

for path in paths:
    text_file = open(path+'filenames.txt', 'r')
    lines = text_file.read().splitlines()

    for name in lines:
        img = cv2.imread(path+name)
        img = cv2.resize(img, (width, height))
        img_original = img
        cropped = img[0:height, center_x:center_x+1]

        # STEPS TO CHANGE SATURATION
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV).astype("float32")
        (h,s,v) = cv2.split(hsv)
        # s is a 1D array containing the values for saturation
        s = np.clip(s,0,255)
        # s_avg first has the some of every array s and then
        # gets the average by dividing it by the # of images
        # outside the inner loop

        sat_sum = sum(s)
        s_avg = sat_sum / len(s)
        if path == liquid_path:
            print "Liquid: "+str(s_avg)
            liquid.append(s_avg)
        elif path == empty_path:
            print "Empty: "+str(s_avg)
            empty.append(s_avg)
        else:
            print "Unknown: "+str(s_avg)
            unknown.append(s_avg)
avg_liquid = sum(liquid) / len(liquid)
avg_empty = sum(empty) / len(empty)
avg_unknown = sum(unknown) / len(unknown)

print "Average liquid = "+str(avg_liquid)
print "Average  empty = "+str(avg_empty)
print "Average unknown = "+str(avg_unknown)

liquid.sort()
empty.sort()
unknown.sort()



plt.figure(1)
plt.title("Plot for saturation")
plt.plot(liquid, label = "Liquid")
plt.plot(empty, label = "Empty")
plt.plot(unknown, label = "Unknown")
plt.xlabel("Img ID")
plt.ylabel("Saturation value")
plt.show()

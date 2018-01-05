import numpy as np
import matplotlib.pyplot as plt
import cv2

liquid_path  = "data/Liquid/"
empty_path   = "data/Empty/"
unknown_path = "data/Unknown/"

paths = [liquid_path, empty_path, unknown_path]

width = height = 400
center_x = int(width / 2)

s_avg = np.array([400, 1])

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
        s = np.clip(s,0,255)
        s_avg = s_avg + s

    s_avg = s_avg / len(lines)
    results.append(s_avg)

plt.figure(1)
plt.title("Plot for saturation")
plt.plot(results[0], label = "Liquid")
plt.plot(results[1], label = "Empty")
plt.plot(results[2], label = "Unknown")
plt.xlabel("Pixel position")
plt.ylabel("Saturation value")
plt.legend()
plt.show()

import numpy as np
import cv2

def getAverage(lines, path):
    avg_height = 0
    avg_width  = 0
    for name in lines:
        img_name = name
        # Load a color image
        img = cv2.imread(path+img_name)

        height, width = img.shape[:2]

        avg_height = avg_height + height
        avg_width = avg_width + width
    avg_height = int(avg_height / len(lines))
    avg_width = int(avg_width / len(lines))

    answer = [avg_width, avg_height]
    print "Width and height: ", answer
    return answer

# Read the name of all available images into `lines`
liquid_path  = "data/Liquid/"
empty_path   = "data/Empty/"
unknown_path = "data/Unknown/"
text_file = open(unknown_path+'filenames.txt', 'r')
lines = text_file.read().splitlines()

average = getAverage(lines, unknown_path)
w = average[0]
h = average[1]
center_x = int(w / 2)
tmp_width = len(lines)+1
widthIndex = 0

tmp = np.zeros((h,tmp_width,3), np.uint8)

# Testing Copy & Write
for name in lines:
    img_name = name
    img = cv2.imread(unknown_path+img_name)
    resized_image = cv2.resize(img, (w, h))
    #resized_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    tmp[0:h, widthIndex:widthIndex+1] = resized_image[0:h, center_x:center_x+1]
    widthIndex = widthIndex + 1

#cv2.imshow('image', resized_image)
#cv2.imshow('image2', img)
print "Printing image..."
tmp = cv2.GaussianBlur(tmp, (7, 7), 0)
cv2.imshow('tmp', tmp)
print "Saving image..."
cv2.imwrite('unknown_graph_gaussian_pos.jpg', tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()
print "Done."

import numpy as np
import cv2

def getDimensionAverage(lines, path):
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

def getPixelStripeAverage(tmp):
    height, width = tmp.shape[:2]
    print height, width
    # Create the image to receive the average colors
    average_img = np.zeros((height, width, 3), np.uint8)

    blue = green = red = 0

    # Loop through to get the colors
    for hi in range(0, height):
        for wi in range(0, width):
            px    = tmp[hi,wi]
            blue  = blue  + px[0]
            green = green + px[1]
            red   = red   + px[2]
        blue  = int(blue / width)
        green = int(green / width)
        red   = int(red / width)
        # Draw line with the averaged color
        cv2.line(average_img, (0, hi), (width-1, hi), (blue, green, red), 1)
        blue = green = red = 0
    return average_img

# Read the name of all available images into `lines`
liquid_path  = "data/Liquid/"
empty_path   = "data/Empty/"
unknown_path = "data/Unknown/"

# Set final path
final_path = unknown_path

text_file = open(final_path+'filenames.txt', 'r')
lines = text_file.read().splitlines()

# Get dimension average to resize all images
    #average = getDimensionAverage(lines, final_path)
    #w = average[0]
    #h = average[1]

h = 400
w = 400
center_x = int(w / 2)
tmp_width = len(lines)+1
widthIndex = 0

tmp = np.zeros((h,tmp_width,3), np.uint8)

# Testing Copy & Write
for name in lines:
    img_name = name
    img = cv2.imread(final_path+img_name)
    resized_image = cv2.resize(img, (w, h))
    #resized_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    tmp[0:h, widthIndex:widthIndex+1] = resized_image[0:h, center_x:center_x+1]
    widthIndex = widthIndex + 1

#cv2.imshow('image', resized_image)
#cv2.imshow('image2', img)
print "Printing image..."
#tmp = cv2.GaussianBlur(tmp, (7, 7), 0)
cv2.imshow('tmp', tmp)

averaged_img = getPixelStripeAverage(tmp)
cv2.imshow("generated_image", averaged_img)
print "Saving image..."
cv2.imwrite('unknown_averaged_graph_400h.jpg', averaged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print "Done."

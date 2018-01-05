import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the name of all available images into `lines`
liquid_path  = "dataIMN/Liquid/"
empty_path   = "dataIMN/Empty/"
unknown_path = "dataIMN/Unknown/"

#paths = [liquid_path, empty_path, unknown_path]
paths = [empty_path, unknown_path]

for path in paths:
    final_path = path
    text_file = open(final_path+'filenames.txt', 'r')
    lines = text_file.read().splitlines()

    for name in lines:
        #img_name = lines[212]
        img = cv2.imread(final_path+name)

        width = 400
        height = 400
        center_x = int(width / 2)
        img = cv2.resize(img, (width, height))
        img_original = img
        cropped = img[0:height, center_x:center_x+1]

        # Creating mask
        mask = np.zeros(img.shape[:2], np.uint8)

        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        rect = (3, 3, 396, 396)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        cv2.imshow("img", img)
        #plt.imshow(img), plt.colorbar(), plt.show()
        cv2.waitKey(1)
        impath = "grabcut/"+name
        cv2.imwrite(impath, img)
        print "Saving "+name+" into "+impath
cv2.destroyAllWindows()

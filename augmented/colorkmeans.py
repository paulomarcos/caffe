import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils


# Print the histogram
# In essence, all this function is doing is counting
# the number of pixels that belong to each cluster
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_))+1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # Normalize the histogram such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

# Generates a figure displaying how many pixels were
# assigned to each cluster based on the output of the
# centroid_histogram function.
def plot_colors(hist, centroids):
    # Initialize the bar chart representing the
    # relative frequency of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # Loop over the percentage of each cluster and the
    # color of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

# Read the name of all available images into `lines`
liquid_path  = "data/Liquid/"
empty_path   = "data/Empty/"
unknown_path = "data/Unknown/"
image_path = empty_path

text_file = open(image_path+'filenames.txt', 'r')
lines = text_file.read().splitlines()

img_name = lines[46]
img = cv2.imread(image_path+img_name)

img = cv2.resize(img, (400, 400))

# Convert to RGB so that we can display it with matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show our image
plt.figure()
plt.axis("off")
plt.imshow(img)

# Reshape the image to be a list of pixels
img = img.reshape((img.shape[0] * img.shape[1], 3))

# Cluster the pixel intensities
clt = KMeans(n_clusters = 6)
clt.fit(img)

# Build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)

# Show bar
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
plt.pause(0)
plt.draw()

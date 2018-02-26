import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe
import sys
import os, errno
from termcolor import colored


def getState(image_name):
	if image_name[:9] == "n01111111":
		return "liquid"
	elif image_name[:9] == "n01122222":
		return "empty"
	elif image_name[:9] == "n01133333":
		return "unknown"
	else:
		return "error"

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../models/bvlc_googlenet/deploy.prototxt'#'../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_googlenet/bvlc_googlenet_iter_900000.caffemodel'#'../models/bvlc_reference_caffenet/caffenet_train_iter_450000.caffemodel'

# load the model
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
					   mean=np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
					   channel_swap=(2,1,0),
					   raw_scale=255,
					   image_dims=(256, 256))
print "successfully loaded classifier"

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)


PATH_PREFIX = '/home/gother/caffe-cupstate/data/ilsvrc12/'
IMAGE_PREFIX = PATH_PREFIX+"val/"

lines = []
with open(PATH_PREFIX+"filenames.txt") as f:
	for line in f:
		line = line.strip()
		lines.append(line)

# load ImageNet labels
labels_file = PATH_PREFIX + 'synset_words.txt'
if not os.path.exists(labels_file):
    print "ErRoR: path doesnt exist!!!"

labels = np.loadtxt(labels_file, str, delimiter='\t')

# sort top five predictions from softmax output
#top_inds = out_prob.argsort()[::-1][:3]  # reverse sort and take three largest items

#print 'probabilities and labels:', zip(out_prob[top_inds], labels[top_inds])

wrong_guesses = []
right_guesses = []

for image_name in lines:
	image = caffe.io.load_image(IMAGE_PREFIX+image_name)
	pred = net.predict([image])
	net.blobs['data'].data[...] = transformer.preprocess('data', image)

	out = net.forward()
	out_prob = out['prob'][0]
	result = out['prob'].argmax()
	# sort top three predictions from softmax output
	top_inds = out_prob.argsort()[::-1][:1]

	if getState(image_name) == "liquid":
		if result == 0:
			right_guesses.append(out_prob[top_inds][0])
			print colored('Right guess:', 'green'), zip(out_prob[top_inds], labels[top_inds])
		else:
			wrong_guesses.append(out_prob[top_inds][0])
			print colored('Wrong guess:', 'red'), zip(out_prob[top_inds], labels[top_inds])
	if getState(image_name) == "empty":
		if result == 1:
			right_guesses.append(out_prob[top_inds][0])
			print colored('Right guess:', 'green'), zip(out_prob[top_inds], labels[top_inds])
		else:
			wrong_guesses.append(out_prob[top_inds][0])
			print colored('Wrong guess:', 'red'), zip(out_prob[top_inds], labels[top_inds])
	if getState(image_name) == "unknown":
		if result == 2:
			right_guesses.append(out_prob[top_inds][0])
			print colored('Right guess:', 'green'), zip(out_prob[top_inds], labels[top_inds])
		else:
			wrong_guesses.append(out_prob[top_inds][0])
			print colored('Wrong guess:', 'red'), zip(out_prob[top_inds], labels[top_inds])

plt.figure(1)
plt.title("Plot for saturation")
plt.plot(sorted(right_guesses), label = "Right")
plt.plot(sorted(wrong_guesses), label = "Wrong")
plt.xlabel("Guess id")
plt.ylabel("Guess score")
plt.legend()
plt.show()

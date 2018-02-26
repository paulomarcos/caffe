import numpy as np
import cv2
import os, errno
import matplotlib.pyplot as plt
from PIL import Image
import sys
import caffe
#import opencv

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

answer = []

PATH_PREFIX = '/home/gother/caffe-cupstate/data/ilsvrc12/'
IMAGE_PREFIX = PATH_PREFIX+"val/"

lines = []
with open(PATH_PREFIX+"filenames.txt") as f:
	for line in f:
		line = line.strip()
		lines.append(line)

right_score_L = 0
right_score_E = 0
right_score_U = 0
wrong_score_L = 0
wrong_score_E = 0
wrong_score_U = 0
guess_L = 0
guess_E = 0
guess_U = 0
Liquid_wrong_E = 0
Liquid_wrong_U = 0
Empty_wrong_L = 0
Empty_wrong_U = 0
Unknown_wrong_L = 0
Unknown_wrong_E = 0

directory = "failed_classification_googlenet_900000_combined"
try:
	os.makedirs(directory)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

green = (10, 90, 10)
red = (10, 10, 235)

# test on a image
for n in lines:
	IMAGE_FILE = IMAGE_PREFIX+n
	input_image = caffe.io.load_image(IMAGE_FILE)
	# predict takes any number of images,
	# and formats them for the Caffe net automatically
	pred = net.predict([input_image])
	net.blobs['data'].data[...] = transformer.preprocess('data', input_image)

	img = cv2.imread(IMAGE_FILE)

	#compute
	out = net.forward()

	#print "Predicted class: "
	#print out['prob'].argmax()

	result = out['prob'].argmax()

	if n.find("n01111111") >= 0:
		if result == 0:
			right_score_L = right_score_L + 1
			font_text = "Liquid"
			color = green
			#print n+" - L R"
		else:
			color = red
			if (result == 1):
				font_text = "Empty"
				Liquid_wrong_E = Liquid_wrong_E + 1
			elif (result == 2):
				font_text = "Unknown"
				Liquid_wrong_U = Liquid_wrong_U + 1
			wrong_score_L = wrong_score_L + 1
			#print n+" - L W"
	if n.find("n01122222") >= 0:
		if result == 1:
			font_text = "Empty"
			right_score_E = right_score_E + 1
			color = green
			#print n+" - E R"
		else:
			color = red
			if (result == 0):
				font_text = "Liquid"
				Empty_wrong_L = Empty_wrong_L + 1
			elif (result == 2):
				font_text = "Unknown"
				Empty_wrong_U = Empty_wrong_U + 1
			wrong_score_E = wrong_score_E + 1
			#print n+" - E W"
	if n.find("n01133333") >= 0:
		if result == 2:
			font_text = "Unknown"
			right_score_U = right_score_U + 1
			color = green
			#print n+" - U R"
		else:
			color = red
			if (result == 1):
				font_text = "Empty"
				Unknown_wrong_E = Unknown_wrong_E + 1
			elif (result == 0):
				font_text = "Liquid"
				Unknown_wrong_L = Unknown_wrong_L + 1
			wrong_score_U = wrong_score_U + 1
			#print n+" - U W"

	cv2.rectangle(img,(9,11),(96,39),(155,155,155),-1)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img,font_text,(16,30), font, 0.55,color,1,cv2.LINE_AA)
	if (color == red):
		cv2.imwrite(directory+"/"+n, img)

	if (result == 0):
		guess_L = guess_L + 1
	elif (result == 1):
		guess_E = guess_E + 1
	elif (result == 2):
		guess_U = guess_U + 1

	answer.append(result)

	#print predicted labels
	labels = np.loadtxt("../data/ilsvrc12/synset_words.txt", str, delimiter='\t')
	top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
	#print labels[top_k]


"""print "Final answer: "
l=1
for n in answer:
	if n ==0:
		c = "Liquid"
	if n == 1:
		c = "Empty"
	if n == 2:
		c = "Unknown/Opaque"
	print str(l)+" -> "+c
	l = l+1
"""
print "\nRight score | Wrong score (Liquid):"
print str(right_score_L)+" | "+str(wrong_score_L)
print "Accuracy Liquid:"
print str(right_score_L*100/(right_score_L+wrong_score_L))+"%"
print "Wrong Guesses:"
print "Empty: "+str(Liquid_wrong_E)+" | Unknown: "+str(Liquid_wrong_U)

print "\nRight score | Wrong score (Empty):"
print str(right_score_E)+" | "+str(wrong_score_E)
print "Accuracy Empty:"
print str(right_score_E*100/(right_score_E+wrong_score_E))+"%"
print "Wrong Guesses:"
print "Liquid: "+str(Empty_wrong_L)+" | Unknown: "+str(Empty_wrong_U)

print "\nRight score | Wrong score (Unknown):"
print str(right_score_U)+" | "+str(wrong_score_U)
print "Accuracy Unknown:"
print str(right_score_U*100/(right_score_U+wrong_score_U))+"%"
print "Wrong Guesses:"
print "Liquid: "+str(Unknown_wrong_L)+" | Empty: "+str(Unknown_wrong_E)

right_score = right_score_L + right_score_E + right_score_U
wrong_score = wrong_score_L + wrong_score_E + wrong_score_U
print "\nTotal Accuracy:"
print str(right_score*100/(right_score+wrong_score))+"%"

print "\nTotal Guesses Liquid: "
print guess_L
print "Total Guesses Empty: "
print guess_E
print "Total Guesses Unknown: "
print guess_U

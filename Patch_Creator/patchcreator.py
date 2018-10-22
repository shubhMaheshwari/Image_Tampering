import cv2
import numpy as np
import glob
import Tkinter
import tkFileDialog
import os
import util

patch_dimension = 64
MAX_size = 500000

'''
#FOR IEEE FORENSICS DATASET

totalarray =np.zeros([MAX_size,patch_dimension,patch_dimension, 3], dtype=np.uint8)
totallabels = np.zeros([MAX_size, 2], dtype=np.uint8)
savedir = ""


imagedir = "/home/rudrabha/IIITH/ImageTampering/dataset-dist/phase-01/training/fake/"
maskdir = "mask"
totalcnt = 0
c=0		
for img_from_folder in glob.glob(maskdir+"/*.png"):
	img = cv2.imread(img_from_folder,0)
	filename = os.path.basename(img_from_folder)
	img1 = cv2.imread(imagedir+filename)
	height = img.shape[0]
	width = img.shape[1]
	print filename
	patchofanimage, labels, count = util.compareMaskImage(img1, img, height, width)
	if count == -1:
		continue
	c= c+1
	print c
	if totalcnt+count >= MAX_size:
		print "Array size exhausted"
		break
	totalarray[totalcnt:(totalcnt+count)] = patchofanimage 
	totallabels[totalcnt:(totalcnt+count)] = labels 
	totalcnt += count
totalarray = totalarray[0:totalcnt]
totallabels = totallabels[0:totalcnt]	
totalarray = totalarray.astype(np.uint8)
totallabels = totallabels.astype(np.uint8)
np.savez_compressed(savedir+'fakeset_IEEE.npz', images=totalarray, labels=totallabels)
'''

#FOR CASIA DATASET
totalarray =np.zeros([MAX_size,patch_dimension,patch_dimension, 3], dtype=np.uint8)
totallabels = np.zeros([MAX_size, 2], dtype=np.uint8)

maskdir = "../coco/images/mask2014/"
imagedir = "../coco/images/new_train2014/"
savedir = "../coco/images/patch2014"
os.mkdir(savedir)
totalcnt = 0
c = 0
for img_from_folder in glob.glob(maskdir+"/*.tif"):
	img = cv2.imread(img_from_folder,0)
	filename = os.path.basename(img_from_folder)
	img1 = cv2.imread(imagedir+filename)
	height = img.shape[0]
	width = img.shape[1]
	print filename
	patchofanimage, labels, count = util.compareMaskImage(img1, img, height, width)
	if count == -1:
		raw_input("Warning: Image and Mask Shapes did not match")
		continue
	c=c+1
	print c
	
	if totalcnt+count >= MAX_size:
		print "Array size exhausted"
		break
	totalarray[totalcnt:(totalcnt+count)] = patchofanimage 
	totallabels[totalcnt:(totalcnt+count)] = labels 
	totalcnt += count
	print totalcnt

for img_from_folder in glob.glob(maskdir+"/*.jpg"):
	img = cv2.imread(img_from_folder,0)
	filename = os.path.basename(img_from_folder)
	img1 = cv2.imread(imagedir+filename)
	height = img.shape[0]
	width = img.shape[1]
	print filename
	patchofanimage, labels, count = util.compareMaskImage(img1, img, height, width)
	if count == -1:
		raw_input("Warning: Image and Mask Shapes did not match")
		continue

	c=c+1
	print c
	if totalcnt+count >= MAX_size:
		print "Array size exhausted"
		break
	totalarray[totalcnt:(totalcnt+count)] = patchofanimage 
	totallabels[totalcnt:(totalcnt+count)] = labels 
	totalcnt += count
	print totalcnt




totalarray = totalarray[0:totalcnt]
totallabels = totallabels[0:totalcnt]	

np.savez_compressed(savedir+'fakeset_CASIA.npz', images=totalarray, labels=totallabels)







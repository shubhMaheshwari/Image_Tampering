from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import os

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
np.set_printoptions(threshold=np.inf)

# Data
dataDir='../coco/images/train2014/'
# Save the masks 
maskpath = "../coco/images/mask2014/"
# Save the images for using as baseline images
new_train_path = "../coco/images/new_train2014/"
# Inpainting images
inpaintmaskpath = "../coco/images/CVIPmasktrain2014/"

try:
	os.mkdir(maskpath)
	os.mkdir(new_train_path)
	os.mkdir(inpaintmaskpath)
except: 
	pass


dataType='train2014'
annFile='../coco/annotations/instances_train2014.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations
coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['person', 'vehicle', 'outdoor', 'accessory', 'animal', 'sports', 'kitchen', 'furniture', 'food', 'indoor','appliance', 'electronic'])
imgIds = coco.getImgIds(catIds=catIds );

# 20000 for patches 20000 for inpainting using cv inpainting
for i in range(2*20000):
	img = coco.loadImgs(imgIds[i])[0]
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
	anns = coco.loadAnns(annIds)
	ri = np.random.randint(0,len(anns))
	print img['file_name']+" "+str(len(anns))
	mask = coco.annToMask(anns[ri])
	mask = mask*255
	# Save the masks and original image in a different dataset
	filename = img['file_name']

	os.system('cp {} {}'.format(dataDir + filename,new_train_path))
	if i < 20000:
		cv2.imwrite(maskpath+filename ,mask)
	else:
		cv2.imwrite(inpaintmaskpath+filename ,mask)
			

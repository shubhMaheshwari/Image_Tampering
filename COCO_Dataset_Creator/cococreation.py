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
maskpath = "../coco/images/semantic_path/"
# Save the images for using as baseline images
new_train_path = "../coco/images/semantic_normal/"

try:
	os.mkdir(new_train_path)
	os.mkdir(maskpath)
except: 
	pass


dataType='train2014'
annFile='../coco/annotations/instances_train2014.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations
coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['person', 'vehicle', 'outdoor', 'accessory', 'animal', 'sports', 'kitchen', 'furniture', 'food', 'indoor','appliance', 'electronic'])
imgIds = coco.getImgIds(catIds=catIds );

# 20000 for patches 20000 for inpainting using cv inpainting
cnt = 0
i = 0
while True:
	img = coco.loadImgs(imgIds[i])[0]
	i+= 1		
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
	anns = coco.loadAnns(annIds)
	ri = np.random.randint(0,len(anns))
	print img['file_name']+" "+str(len(anns))
	mask = coco.annToMask(anns[ri])

	if(np.mean(mask) < 0.03 or np.mean(mask) > 0.2):
		continue
	else:
		cnt+=1

	mask = mask*255
	# Save the masks and original image in a different dataset
	filename = img['file_name']

	os.system('cp {} {}'.format(dataDir + filename,new_train_path))
	cv2.imwrite(maskpath+filename ,mask)
	
	if cnt > 20000:
		break
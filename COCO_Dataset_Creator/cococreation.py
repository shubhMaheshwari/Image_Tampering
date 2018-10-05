from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
np.set_printoptions(threshold=np.inf)
maskpath = "mask2014/"
dataDir='train2014'
dataType='train2014'
annFile='/home/rudrabha/IIITH/ImageTampering/annotations/instances_train2014.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations
coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['person', 'vehicle', 'outdoor', 'accessory', 'animal', 'sports', 'kitchen', 'furniture', 'food', 'indoor','appliance', 'electronic'])
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = imgIds[0])
#img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
for i in range(17000):
	img = coco.loadImgs(imgIds[i])[0]
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
	anns = coco.loadAnns(annIds)
	ri = np.random.randint(0,len(anns))
	print img['file_name']+" "+str(len(anns))
	mask = coco.annToMask(anns[ri])
	cv2.imwrite(maskpath+img['file_name'] ,mask*255)

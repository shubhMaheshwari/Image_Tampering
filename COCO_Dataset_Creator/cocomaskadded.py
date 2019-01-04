import numpy as np
import cv2
import glob
import os

dataDir='../coco/images/'

maskedpath = dataDir + "semantic_mask/"
realpath = dataDir + "semantic_normal/"
maskaddedpath = dataDir + "semantic_mask_added/"

os.mkdir(maskaddedpath)
for img_from_folder in glob.glob(maskedpath+"/*.jpg"):
	I = cv2.imread(img_from_folder)
	filename = os.path.basename(img_from_folder)
	print(filename)
	realI = cv2.imread(realpath+filename)
	realI = cv2.add(realI, I)
	cv2.imwrite(maskaddedpath+filename.split('.')[0] + '.png', realI)


# maskedpath = dataDir + "CVIPmasktrain2014"
# maskaddedpath = dataDir + "CVIPmaskaddedtrain2014/"

# os.mkdir(maskaddedpath)
# for img_from_folder in glob.glob(maskedpath+"/*.jpg"):
# 	I = cv2.imread(img_from_folder)
# 	filename = os.path.basename(img_from_folder)
# 	realI = cv2.imread(realpath+filename)
# 	realI = cv2.add(realI, I)
# 	cv2.imwrite(maskaddedpath+filename, realI)

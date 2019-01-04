import util
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
masked_path = "../CASIA/CASIA2/Masks"

for img_from_folder in glob.glob(masked_path+"/*.tif"):
	
	# f,ax = plt.subplots(1,3)
	I = cv2.imread(img_from_folder)
	# ax[0].imshow(I)
	modifiedmask = util.imfill(I)
	# ax[1].imshow(modifiedmask)
	modifiedmask = util.imageNegative(modifiedmask)
	# ax[2].imshow(modifiedmask)	
	cv2.imwrite(img_from_folder, modifiedmask)
	# plt.show()
for img_from_folder in glob.glob(masked_path+"/*.bmp"):
	I = cv2.imread(img_from_folder)
	modifiedmask = util.imfill(I)
	modifiedmask = util.imageNegative(modifiedmask)
	cv2.imwrite(img_from_folder, modifiedmask)
	
for img_from_folder in glob.glob(masked_path+"/*.jpg"):
	I = cv2.imread(img_from_folder)
	modifiedmask = util.imfill(I)
	modifiedmask = util.imageNegative(modifiedmask)
	cv2.imwrite(img_from_folder, modifiedmask)
	



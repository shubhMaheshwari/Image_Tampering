import util
import cv2
import numpy as np
import os
import glob

masked_path = "../CASIA/CASIA2/Masks/"

for img_from_folder in glob.glob(masked_path+"/*.tif"):
	I = cv2.imread(img_from_folder)
	modifiedmask = util.imfill(I)
	modifiedmask = util.imageNegative(modifiedmask)
	cv2.imwrite(img_from_folder, modifiedmask)
	
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
	



import numpy as np
import cv2
import glob
import os

maskedpath = "mask2014"
realpath = "train2014/"
maskaddedpath = "train2014semantics/"
for img_from_folder in glob.glob(maskedpath+"/*.jpg"):
	I = cv2.imread(img_from_folder)
	filename = os.path.basename(img_from_folder)
	realI = cv2.imread(realpath+filename)
	realI = cv2.add(realI, I)
	cv2.imwrite(maskaddedpath+filename, realI)

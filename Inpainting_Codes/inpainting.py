import numpy as np
import cv2
import glob
import os


maskedpath = "../coco/images/CVIPmasktrain2014/"
maskaddedpath = "../coco/images/CVIPmaskaddedtrain2014/"
savepath = "../coco/images/Inpaintingtrain2014/"

os.mkdir(savepath)
cnt = 0
for img_from_folder in glob.glob(maskaddedpath+"/*.jpg"):
	I = cv2.imread(img_from_folder)
	filename = os.path.basename(img_from_folder)
	maskedI = cv2.imread(maskedpath+filename, 0)
	I = cv2.inpaint(I, maskedI, 3, cv2.INPAINT_TELEA)
	cnt = cnt+1
	print cnt
	cv2.imwrite(savepath+filename, I)

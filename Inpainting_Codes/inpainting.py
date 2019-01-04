import numpy as np
import cv2
import glob
import os


maskedpath = "../coco/images/semantic_mask/"
maskaddedpath = "../coco/images/semantic_mask_added/"
savepath = "../coco/images/semantic_inpainting/"

# os.mkdir(savepath)
cnt = 0
for img_from_folder in glob.glob(maskaddedpath+"/*.png"):
	I = cv2.imread(img_from_folder)
	filename = os.path.basename(img_from_folder)
	maskedI = cv2.imread(maskedpath+filename.split('.')[0] + '.jpg', 0)
	print filename
	I = cv2.inpaint(I, maskedI, 3, cv2.INPAINT_TELEA)
	cnt = cnt+1
	cv2.imwrite(savepath+filename, I)

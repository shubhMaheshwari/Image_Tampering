import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

normal_dir = "../coco/images/semantic_normal/"
new_mask_dir = "../coco/images/patches_mask/"
savepath = "../coco/images/patches_inpainting/"

# os.mkdir(new_mask_dir)
# os.mkdir(savepath)
cnt = 0
for img_from_folder in glob.glob(normal_dir+"/*.jpg"):
	I = cv2.imread(img_from_folder)
	filename = os.path.basename(img_from_folder)
	
	h,w,c = I.shape
	maskedI = np.zeros((h,w),dtype=np.uint8)
	
	x_start = max(int(0.3*np.random.random()*w) , int((np.random.random() - 0.3)*w))
	y_start = max(int(0.3*np.random.random()*h) , int((np.random.random() - 0.3)*h))
	# Each path size size between 0.3*0.3 to 0.7*0.7 of the image
	x_end 	= x_start + int( (0.4*np.random.random() + 0.3)*w)
	y_end   = y_start + int( (0.4*np.random.random() + 0.3)*h)
	maskedI[y_start:y_end, x_start:x_end] = 255
	I = cv2.inpaint(I, maskedI, 3, cv2.INPAINT_TELEA)
	cnt = cnt+1
	filename = filename.split('.')[0] + '.png'
	print cnt,filename
	cv2.imwrite(savepath+filename, I)
	cv2.imwrite(new_mask_dir+filename, maskedI)

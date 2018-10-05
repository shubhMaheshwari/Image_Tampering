import cv2
import os
import glob
import numpy as np
pristine_path = "Au/"
tampered_path = "Tp"
masked_path = "Masks/"

for img_from_folder in glob.glob(tampered_path+"/*.tif"):
	I = cv2.imread(img_from_folder)
	filename = os.path.basename(img_from_folder)
	print filename
	bgimgname = filename.split("_")[5]
	bgimgname = bgimgname[0:3]+"_"+bgimgname[3:8]
	filename1 = pristine_path+"Au_"+bgimgname+".jpg"
	print filename1
	Ib = cv2.imread(filename1)
	Ib = np.asarray(Ib)
	if Ib.size == 0:
		filename1 = pristine_path+"Au_"+bgimgname+".JPG"
		Ib = cv2.imread(filename1)
		
	if I.shape != Ib.shape:
		continue
	Id = cv2.subtract(I,Ib)
	Id = cv2.cvtColor(Id, cv2.COLOR_RGB2GRAY)
	ret,thresh1 = cv2.threshold(Id,20,255,cv2.THRESH_BINARY)
	cv2.imwrite(masked_path+filename, thresh1)
	
for img_from_folder in glob.glob(tampered_path+"/*.bmp"):
	I = cv2.imread(img_from_folder)
	filename = os.path.basename(img_from_folder)
	print filename
	bgimgname = filename.split("_")[5]
	bgimgname = bgimgname[0:3]+"_"+bgimgname[3:8]
	filename1 = pristine_path+"Au_"+bgimgname+".jpg"
	print filename1
	Ib = cv2.imread(filename1)
	Ib = np.asarray(Ib)
	if Ib.size == 0:
		filename1 = pristine_path+"Au_"+bgimgname+".JPG"
		Ib = cv2.imread(filename1)
	if I.shape != Ib.shape:
		continue
	Id = cv2.subtract(I,Ib)
	Id = cv2.cvtColor(Id, cv2.COLOR_RGB2GRAY)
	ret,thresh1 = cv2.threshold(Id,20,255,cv2.THRESH_BINARY)
	cv2.imwrite(masked_path+filename, thresh1)
	
for img_from_folder in glob.glob(tampered_path+"/*.jpg"):
	I = cv2.imread(img_from_folder)
	filename = os.path.basename(img_from_folder)
	print filename
	bgimgname = filename.split("_")[5]
	bgimgname = bgimgname[0:3]+"_"+bgimgname[3:8]
	filename1 = pristine_path+"Au_"+bgimgname+".jpg"
	print filename1
	Ib = cv2.imread(filename1)
	Ib = np.asarray(Ib)
	if Ib.size == 0:
		filename1 = pristine_path+"Au_"+bgimgname+".JPG"
		Ib = cv2.imread(filename1)
	if I.shape != Ib.shape:
		continue
	Id = cv2.subtract(I,Ib)
	Id = cv2.cvtColor(Id, cv2.COLOR_RGB2GRAY)
	ret,thresh1 = cv2.threshold(Id,20,255,cv2.THRESH_BINARY)
	cv2.imwrite(masked_path+filename, thresh1)
	



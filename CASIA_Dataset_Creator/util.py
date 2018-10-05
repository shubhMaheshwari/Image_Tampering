import cv2;
import numpy as np;

 
def imfill(im_in):
	kernel = np.ones((5,5),np.uint8)
	im_copy = im_in.copy()
	thresh = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)

	im,contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	cv2.drawContours(im_copy,contours,-1,(255,255,255), 5)
	closing = cv2.morphologyEx(im_copy, cv2.MORPH_CLOSE, kernel)
	return closing

def imageNegative(im):
	im = cv2.bitwise_not(im)
	return im

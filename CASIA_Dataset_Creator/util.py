import cv2;
import numpy as np;

 
kernel = np.array([ [0, 0, 1, 0, 0],
			[0, 1, 1, 1, 0],
			[1, 1, 1, 1, 1],
			[0, 1, 1, 1, 0],
			[0, 0, 1, 0, 0]], dtype=np.uint8)
# kernel = np.ones((5,5),dtype=np.uint8)
def imfill(im_in):
	im_copy = im_in.copy()
	im_copy2 = im_in.copy()
	thresh = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)

	im,contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	cv2.drawContours(im_copy,contours,-1,(255,255,255), 1)
	closing = cv2.morphologyEx(im_copy, cv2.MORPH_CLOSE, kernel,iterations = 1)
	closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)

	im,contours, hierarchy= cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	cv2.drawContours(im_copy2,contours,-1,(255,255,255), 1)
	closing = cv2.morphologyEx(im_copy2, cv2.MORPH_CLOSE, kernel,iterations = 1)
	closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
	return closing

def imageNegative(im):
	im = cv2.bitwise_not(im)
	return im

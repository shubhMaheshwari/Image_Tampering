import cv2
import numpy as np
import glob
import os
import util
import testing
import random

patch_dimension = 64
MAX_size = 50000
imagedir = "../../CASIA/CASIA1/CASIA1/Au/"

def getRandomFile(path):
	"""
	Returns a random filename, chosen among the files of the given path.
	"""
	files = os.listdir(path)
	index = random.randrange(0, len(files))
	return files[index]

pristinecnt = 0
tamperedcnt = 0
loopv = 1400

for i in range(loopv):
	fn = getRandomFile(imagedir)
	totalarray =np.zeros([MAX_size,patch_dimension,patch_dimension, 3], dtype=np.uint8)
	jpegcnt = 0
	tifcnt = 0
	totalcnt = 0
	c = 0
	img = cv2.imread(imagedir+fn)
	tifcnt = tifcnt+1
	height = img.shape[0]
	width = img.shape[1]
	patchofanimage, count = util.getallPatchesColor(img, height, width)
	if count == -1:
		raw_input("Warning: Image and Mask Shapes did not match")
	
	c=c+1
	#print c

	if totalcnt+count >= MAX_size:
		print "Array size exhausted"
	
	totalarray[totalcnt:(totalcnt+count)] = patchofanimage 

	totalcnt += count
	#print totalcnt
	totalarray = totalarray[0:totalcnt]
	#print totalarray.shape
	flag, pcnt, tcnt = testing.testImgtype(totalarray)
	print "Patch count = "+str(count)+" Tampered patches = "+str(tcnt)
	if flag == 1:
		pristinecnt = pristinecnt+1
	else:
		tamperedcnt = tamperedcnt+1
print pristinecnt
print tamperedcnt
acc = float(pristinecnt)/float(loopv)
print acc





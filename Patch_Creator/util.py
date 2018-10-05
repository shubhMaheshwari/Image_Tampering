import cv2;
import numpy as np;

# In fills holes in any binary image
def imfill(im_in):
	kernel = np.ones((5,5),np.uint8)
	im_copy = im_in.copy()
	thresh = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)

	im,contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	cv2.drawContours(im_copy,contours,-1,(255,255,255), 5)
	closing = cv2.morphologyEx(im_copy, cv2.MORPH_CLOSE, kernel)
	return closing

# Returns the negative of an image
def imageNegative(im):
	im = cv2.bitwise_not(im)
	return im

# Returns an array of all colour patches that can be formed
def getallPatchesColor(image, height, width, patch_dimension=64, overlap=0, MAX_SIZE=5000):
	i=0
	cnt=0
	totalarray =np.zeros([MAX_SIZE,patch_dimension,patch_dimension,3])
	while (i<height):
		j=0
		while (j<width):
			if i+patch_dimension <= height-1 and j+patch_dimension <= width-1:
				rs=i
				re = i+patch_dimension
				cs = j
				ce = j+patch_dimension

			if i+patch_dimension >= height and j+patch_dimension <=width-1:
				rs = height-(patch_dimension)
				re = height
				cs = j
				ce = j+patch_dimension

			if i+patch_dimension <= height-1 and j+patch_dimension >=width:
				rs = i
				re = i+patch_dimension
				cs = width - (patch_dimension)
				ce = width

			if i+patch_dimension >= height and j+patch_dimension >=width:
				rs = height-(patch_dimension)
				re = height
				cs = width - (patch_dimension)
				ce = width

			cropimage = image[rs:re, cs:ce]
			totalarray[cnt] = cropimage
			cnt = cnt+1
			j=j+patch_dimension-overlap
		i=i+patch_dimension-overlap
	totalarray = totalarray[0:cnt]
	return totalarray, cnt

# Returns all possible patches from a single channel image
def getallPatchesGS(image, height, width, patch_dimension=64, overlap=0, MAX_SIZE=5000):
	i=0
	cnt=0
	totalarray =np.zeros([MAX_SIZE,patch_dimension,patch_dimension])
	while (i<height):
		j=0
		while (j<width):
			if i+patch_dimension <= height-1 and j+patch_dimension <= width-1:
				rs=i
				re = i+patch_dimension
				cs = j
				ce = j+patch_dimension

			if i+patch_dimension >= height and j+patch_dimension <=width-1:
				rs = height-(patch_dimension)
				re = height
				cs = j
				ce = j+patch_dimension

			if i+patch_dimension <= height-1 and j+patch_dimension >=width:
				rs = i
				re = i+patch_dimension
				cs = width - (patch_dimension)
				ce = width

			if i+patch_dimension >= height and j+patch_dimension >=width:
				rs = height-(patch_dimension)
				re = height
				cs = width - (patch_dimension)
				ce = width

			cropimage = image[rs:re, cs:ce]
			totalarray[cnt] = cropimage
			cnt = cnt+1
			j=j+patch_dimension-overlap
		i=i+patch_dimension-overlap
	totalarray = totalarray[0:cnt]
	return totalarray, cnt


#Compare Areas
def compareArea(maskpatch, percentage=0.1):
	countwhite = np.count_nonzero(maskpatch==255)
	size = maskpatch.shape[0]*maskpatch.shape[1]
	p = float(countwhite)/float(size)
	if p < percentage:
		return 0
	else:
		#print p
		return 1

	

# Returns all patches which has a percentage of tampering
def compareMaskImage(image, mask, height, width, patch_dimension=64, overlap=0, MAX_SIZE=5000, percentage=0.1):
	if mask.shape[0]!=image.shape[0] or mask.shape[1]!=image.shape[1]:
		print "Incompatible Input"
		return np.empty([2,2]), np.empty([2,2]), -1
	i=0
	cnt=0
	totalarray =np.zeros([MAX_SIZE,patch_dimension,patch_dimension,3])
	labels = np.zeros([MAX_SIZE, 2])
	while (i<height):
		j=0
		while (j<width):
			if i+patch_dimension <= height-1 and j+patch_dimension <= width-1:
				rs=i
				re = i+patch_dimension
				cs = j
				ce = j+patch_dimension
				#print "IF1"

			if i+patch_dimension >= height and j+patch_dimension <=width-1:
				rs = height-(patch_dimension)
				re = height
				cs = j
				ce = j+patch_dimension
				#print "IF2"

			if i+patch_dimension <= height-1 and j+patch_dimension >=width:
				rs = i
				re = i+patch_dimension
				cs = width - (patch_dimension)
				ce = width
				#print "IF3"

			if i+patch_dimension >= height and j+patch_dimension >=width:
				rs = height-(patch_dimension)
				re = height
				cs = width - (patch_dimension)
				ce = width
				#print "IF4"

			cropmask = mask[rs:re, cs:ce]
			if compareArea(cropmask, percentage) == 0:
				j=j+patch_dimension-overlap
				continue
			cropimage = image[rs:re, cs:ce]	
			labels[cnt] = [0,1]	
			totalarray[cnt] = cropimage
			cnt = cnt+1
			j=j+patch_dimension-overlap
		i=i+patch_dimension-overlap
	totalarray = totalarray[0:cnt]
	labels = labels[0:cnt]
	return totalarray, labels, cnt


def getallRandomPatchesGS(image, patch_dimension=64, noofpatches=100):
	totalarray =np.zeros([noofpatches, patch_dimension, patch_dimension], dtype=np.uint8)
	totallabels=np.zeros([noofpatches,2], dtype=np.uint8)
	height = image.shape[0]
	width = image.shape[1]
	for i in range(noofpatches):
		randomrow = np.random.randint(0, height-patch_dimension-1)
		randomcol = np.random.randint(0, width-patch_dimension-1)
		cropimage = image[randomrow:randomrow+patch_dimension, randomcol:randomcol+patch_dimension]
		totalarray[i] = cropimage
		totallabels[i] = [1,0]
	return totalarray, totallabels, noofpatches


def getallRandomPatchesColor(image, patch_dimension=64, noofpatches=100):
	totalarray =np.zeros([noofpatches, patch_dimension, patch_dimension, 3], dtype=np.uint8)
	totallabels=np.zeros([noofpatches,2], dtype=np.uint8)
	height = image.shape[0]
	width = image.shape[1]
	for i in range(noofpatches):
		randomrow = np.random.randint(0, height-patch_dimension-1)
		randomcol = np.random.randint(0, width-patch_dimension-1)
		cropimage = image[randomrow:randomrow+patch_dimension, randomcol:randomcol+patch_dimension]
		totalarray[i] = cropimage
		totallabels[i] = [1,0]
	return totalarray, totallabels, noofpatches	


		

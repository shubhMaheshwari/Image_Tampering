import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from random import shuffle

mode = "Testing"
dataset_name = "CASIA"
fake_dir = "/media/shubh/My Passport/shubh/CASIA2/Tp"
true_dir = "/media/shubh/My Passport/shubh/CASIA2/Au"
mask_dir = "/media/shubh/My Passport/shubh/CASIA2/Masks"
save_dir = "./CASIA_test_patches/"

def get_patches(img, mask,crop_size = 64): 
	#Function to get patches of images. The patch size is 64x64
	##First get indices of rows
	# Assumes input image is of shape C X H X W

	w, h = img.shape[2], img.shape[1]
	w_flag, h_flag = w - crop_size, h - crop_size
	col_idx, row_idx = [], []
	col_c, row_c = 0, 0 
	while col_c < w_flag:
			col_idx.append(col_c)
			col_c += crop_size
	col_idx.append(w_flag)
	while row_c < h_flag:
			row_idx.append(row_c)
			row_c += crop_size
	row_idx.append(h_flag)
	patches = np.zeros((len(col_idx)*len(row_idx), 3, crop_size, crop_size), dtype ='float32')
	count = 0 
	Y = []
	# Create patches
	for y in row_idx:
		for x in col_idx:
			patch = img[:, y:y+crop_size,x:x+crop_size]
			patches[count] = patch
			count += 1
			if mask is None:
				Y.append(0)
				continue

			if 10 * np.sum(mask[y:y+crop_size,x:x+crop_size]) >= 2*crop_size*crop_size and 10 * np.sum(mask[y:y+crop_size,x:x+crop_size]) <= 8 * crop_size*crop_size :
				Y.append(1)
			else:
				Y.append(0)
	return np.array(patches), np.array(Y),(np.ceil(w/64).astype('int'),np.ceil(h/64).astype('int'))



def _get_mask(image_path):

	if dataset_name == "Adobe":
		immask = Image.open(os.path.join(mask_dir,'_'.join(image_path.split('_')[:-1])+'.jpg'))
		immask = np.array(immask)
		ret,immask = cv2.threshold(immask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		immask = immask//255
	elif dataset_name == "IEEE":
		immask = Image.open(os.path.join(mask_dir,image_path[:-4]+'.mask.png')).convert('L')
		immask = np.array(immask)

		ret,immask = cv2.threshold(immask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		immask = 1 - immask//255
	elif dataset_name == "Semantic":
		immask = Image.open(os.path.join(mask_dir,filename.split('.')[0] +'.jpg') ).convert('L')
		immask = np.array(immask)
		immask = immask//255

	elif dataset_name == "Patches":
		immask = Image.open(os.path.join(mask_dir,image_path[:-4]+'.png')).convert('L')
		immask = np.array(immask)
		ret,immask = cv2.threshold(immask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		immask = immask//255

	elif dataset_name == "CASIA":
		immask = Image.open(os.path.join(mask_dir,image_path)).convert('L')
		immask = np.array(immask)
		ret,immask = cv2.threshold(immask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		immask = 1 - immask//255

	if 	np.mean(immask) < 0.005:
		# print("Too small mask")
		return None

	# Get the mask bounding box
	w,h = immask.shape
	row = np.sum(immask,0) 
	col = np.sum(immask,1)
	x1 = (row!=0).argmax(axis=0)
	x2 = h - (np.flipud(row!=0)).argmax(axis=0)
	y1 = (col!=0).argmax(axis=0)
	y2 = w - (np.flipud(col!=0)).argmax(axis=0)
	# print(row,col)
	# print(x1,x2)
	# print(y1,y2)
	# print(dataset_name,image_path)

	# print("Threshold",ret)
	# print(immask.shape)
	# plt.imshow(immask,cmap='gray')
	# plt.show()
	return (immask,(y1,y2),(x1,x2))


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

cnt = 0
file_list = os.listdir(fake_dir)
shuffle(file_list)
for fileno,filename in enumerate(file_list):
	print(filename)
	image = cv2.imread(os.path.join(fake_dir,filename))

	try:
		mask,y_box,x_box = _get_mask(filename)
		print(image.shape,mask.shape)
	except:
		continue
	# Save sum number or elements
	try:
		patches, Y, patch_size = get_patches(image.T, mask.T)
		print(patches.shape,len(Y),patch_size)
	except:
		print("Patch too small")
		continue
	

	# plt.imshow(mask)
	# plt.show()

	if mode=="Training":
		perm = np.random.permutation(patches.shape[0])
		patches = patches[perm,:,:,:]
		Y = Y[perm]

		h,w = patch_size

		white_count = 0
		patch_cnt = 0
		prob = np.sum(Y)
		print(prob)
		for k in range(patches.shape[0]):
			if white_count < prob or Y[k] == 1: 
				save_name = "{}_{}_{}_{}_{}_{}_{}.png".format(dataset_name,fileno,patch_cnt//h,patch_cnt%h, h,w ,Y[k])
				cv2.imwrite(os.path.join(save_dir,save_name),patches[k,:,:,:].T) 
				cnt+=1
				white_count+=1
				patch_cnt += 1
				print(fileno, cnt, Y[k])

		if cnt > 20000:
			break

	else:

		patch_cnt = 0
		h,w = patch_size
		print(h*w)
		if h*w > 300:
			continue


		for k in range(patches.shape[0]):
			save_name = "{}_{}_{}_{}_{}_{}_{}.png".format(dataset_name,fileno,patch_cnt//h,patch_cnt%h, h,w ,Y[k])
			cv2.imwrite(os.path.join(save_dir,save_name),patches[k,:,:,:].T) 
			patch_cnt+=1
			print(save_name)
		if fileno > 100:
			break


true_list = os.listdir(true_dir)
shuffle(true_list)
for fileno,filename in enumerate(true_list):
	print(filename)
	image = cv2.imread(os.path.join(true_dir,filename))
	patches, Y, patch_size = get_patches(image.T, None)
	print(patches.shape,len(Y),patch_size)
	
	# print(image.shape)
	h, w  = patch_size
	patch_cnt = 0
	for k in range(patches.shape[0]):
		save_name = "{}_{}_{}_{}_{}_{}_{}.png".format(dataset_name,len(file_list) + fileno,patch_cnt//h,patch_cnt%h, h,w ,Y[k])
		cv2.imwrite(os.path.join(save_dir,save_name),patches[k,:,:,:].T) 
		patch_cnt+=1
		print(save_name)
	if fileno > 100:
		break 

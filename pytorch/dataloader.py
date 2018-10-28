import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import cv2

train_transforms =  transforms.Compose([
	# transforms.RandomHorizontalFlip(),
	# transforms.RandomVerticalFlip(),
	transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

	# transforms.Normalize((0.5, 0.47, 0.45), (0.0262, 0.0249, 0.0261))
])


class DataSet(torch.utils.data.Dataset):
	"""
		Our dataset loader for each training, testing, val dataset

	"""
	def __init__(self,opt,true_images, fake_images,mask_images,dataset):
		super().__init__()
		# Directories
		self.true_dir = true_images
		self.fake_dir = fake_images
		self.mask_dir = mask_images
		self.dataset = dataset
		# Calculate filename list 
		true_list = [ filename for filename in os.listdir(self.true_dir)]		
		fake_list = [ filename for filename in os.listdir(self.fake_dir)]		

		# Make sure fake and true list of the same size
		if len(true_list) > len(fake_list):
			self.true_list = true_list[:len(fake_list)]
			self.fake_list = fake_list
		else:
			self.true_list = true_list
			self.fake_list = fake_list[:len(true_list)]

		self.opt = opt
		print("Dataset size True Images(0):{} Fake Images(1):{}".format(len(self.true_list),len(self.fake_list)))
	
	def _get_mask(self,image_path):

		if self.dataset == "Adobe_Featuring":
			immask = Image.open(os.path.join(self.mask_dir,'_'.join(image_path.split('/')[-1].split('_')[:-1])+'.jpg'))
			immask = np.array(immask)
			ret,immask = cv2.threshold(immask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			immask = immask//255
		elif self.dataset == "IEEE":
			immask = Image.open(os.path.join(self.mask_dir,image_path.split('/')[-1][:-4]+'.mask.png')).convert('L')
			immask = np.array(immask)

			ret,immask = cv2.threshold(immask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			immask = 1 - immask//255
		elif self.dataset == "Segment":
			immask = Image.open(os.path.join(self.mask_dir,image_path.split('/')[-1][:-4]+'.jpg')).convert('L')
			immask = np.array(immask)
			immask = immask//255

		elif self.dataset == "Inpainting":
			immask = Image.open(os.path.join(self.mask_dir,image_path.split('/')[-1][:-4]+'.jpg')).convert('L')
			immask = np.array(immask)
			ret,immask = cv2.threshold(immask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			immask = immask//255




		if 	np.mean(immask) < 0.005:
			# print("Too small mask")
			return None
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
		# print(self.dataset,image_path)

		# print("Threshold",ret)
		# print(immask.shape)
		# plt.imshow(immask,cmap='gray')
		# plt.show()
		return (immask,(y1,y2),(x1,x2))

	def __getitem__(self,idx):
	
		"""
			Given an index it returns the image and its ground truth (fake: 1, true: 0)
		"""
		# idx = idx % 100 
		# if np.random.random() > 0.5:
		# 	idx = idx +  len(self.true_list)
		if idx >= len(self.true_list):
			idx = idx - len(self.true_list)
			try:
				path = os.path.join(self.fake_dir, self.fake_list[idx])
				im = Image.open(path).convert('RGB')
			except Exception as e:
				os.remove(path)
				path = os.path.join(self.fake_dir, self.fake_list[idx+1])
				im = Image.open(path).convert('RGB')
			# print(im.size)
			im = train_transforms(im)
			mask = self._get_mask(path)
			if mask is None: 
				y = 0
			else:
				y = 1


			# plt.subplot(1,2,1)
			# plt.imshow(im.numpy().T)
			# plt.title("{} {} {}".format(self.dataset,y,"Fake"))
			# plt.subplot(1,2,2)
			# if mask is not  None:
			# 	plt.imshow(mask[0].T)
			# plt.show()
	
		else:
			path = os.path.join(self.true_dir, self.true_list[idx])
			im = Image.open(path).convert('RGB')
			im = train_transforms(im)
			y = 0		
			mask = None

			# plt.subplot(1,2,1)
			# plt.imshow(im.numpy().T)
			# plt.title("{} {} {}".format(self.dataset,y,"True"))
			# plt.subplot(1,2,2)
			# if mask is not None:
			# 	plt.imshow(mask[0].T)
			# plt.show()


		

		return im,mask,y

	def __len__(self):
		return len(self.true_list) + len(self.fake_list)

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
	for x in col_idx:
			for y in row_idx:
				patch = img[:, y:y+crop_size,x:x+crop_size]
				if mask is None:
					Y.append(0)
					continue
				# print("Percent ",np.sum(mask[y:y+crop_size,x:x+crop_size])/crop_size**2)
				if 10 * np.sum(mask[y:y+crop_size,x:x+crop_size]) >= 1*crop_size*crop_size and 10 * np.sum(mask[y:y+crop_size,x:x+crop_size]) <= 9 * crop_size*crop_size :
					# print(h,w,mask.shape,img.shape,patches[count,:,:,:].shape)

					Y.append(1)
				else:
					Y.append(0)
				patches[count] = patch
				count += 1
				# if Y[count-1] == 1:
				# 	plt.subplot(1,3,1)
				# 	plt.imshow(img.numpy().T)
				# 	plt.subplot(1,3,2)
				# 	plt.imshow(mask.T)
				# 	plt.subplot(1,3,3)
				# 	plt.imshow(patches[count-1,:,:,:].T)
				# 	plt.title(str(x)+""+str(y))
				# 	plt.show()

	# print("Total Pactches:",count)
	return patches, Y



def collate_fn(data,num_crops=4,crop_size=64):
	"""
	Creates mini-batch tensors from the list of tuples (image, caption).
	We will have multiple images of people for each datapoint
	Args:
		data: list of tuple (image, persons). 
			- image: torch tensor of shape (3, 256, 256).
			- persons: persons detected in the image.
		
		num_crops: 
			- Number of cropped images per image
			- Default = 4
		crop_size:
			- Size of each image			
	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		person_images: torch tensor of total persons present over all images
		target_ind = image_index of original for each person_image's output
	"""
	# Sort a data list by caption length (descending order).
	im,mask_list, Y = zip(*data)


	# Merge images (from tuple of 3D tensor to 4D tensor).
	cropped_im = []
	cropped_y = []
	percent_list = []
	if num_crops>0:
		for i,image in enumerate(im):	
				
			c,h,w = image.shape
			if Y[i]==0: 
				for k in range(num_crops):
					y = np.random.randint(0, h - crop_size + 1)
					x = np.random.randint(0, w - crop_size + 1)
					cropped_im.append(image[:,y:y+crop_size,x:x+crop_size])
					# print("True:",Y[i])
					cropped_y.append(Y[i])
					# plt.subplot(1,2,1)
					# plt.imshow(cropped_im[-1].numpy().T)
					# plt.subplot(1,2,2)
					# plt.imshow(image.numpy().T)
					# plt.title("Sample Cropped:{} y:{}".format(cropped_im[-1].shape,Y[i]))
					# plt.show()
					percent_list.append(0)




			else:
				for k in range(num_crops):
					mask,(y1,y2),(x1,x2) = mask_list[i]
					for rr in range(100000):

						if rr > 100000 -3:
							print("Too much bt")
							x = 0 
							y = 0
							cropped_im.append(image[:,y:y+crop_size,x:x+crop_size])
							if 10*np.sum(mask[y:y+crop_size,x:x+crop_size]) > 2*crop_size*crop_size:
								cropped_y.append(1)
							else:
								cropped_y.append(0)
							percent_list.append(np.sum(mask[y:y+crop_size,x:x+crop_size])/crop_size**2)
							break
						try:	
							x = np.random.randint(max(0,x1-crop_size), min(x2+crop_size,w - crop_size + 1))
							y = np.random.randint(max(0,y1-crop_size), min(y2+crop_size,h - crop_size + 1))
						except Exception as e:
							print(x1,x2,y1,y2,w,h)
							print(max(0,x1-crop_size),min(x2+crop_size,w - crop_size + 1))
							print(max(0,y1-crop_size),min(y2+crop_size,h - crop_size + 1))
							print("Error while getting tampered image for training",e)
							x = 0
							y = 0
							cropped_im.append(image[:,y:y+crop_size,x:x+crop_size])
							if 10*np.sum(mask[y:y+crop_size,x:x+crop_size]) > 2*crop_size*crop_size:
								cropped_y.append(1)
							else:
								cropped_y.append(0)
							percent_list.append(np.sum(mask[y:y+crop_size,x:x+crop_size])/crop_size**2)
							break
						# print(mask.shape,image.shape)
						if 10 * np.sum(mask[y:y+crop_size,x:x+crop_size]) >= 2*crop_size*crop_size and 10 * np.sum(mask[y:y+crop_size,x:x+crop_size]) <= 8 * crop_size*crop_size :
							cropped_im.append(image[:,y:y+crop_size,x:x+crop_size])
							cropped_y.append(Y[i])
							# print("Fake:",Y[i])

							percent_list.append(np.sum(mask[y:y+crop_size,x:x+crop_size])/crop_size**2)
							break
							# else:
							# 	print("Percent ",np.sum(mask[y:y+crop_size,x:x+crop_size])/crop_size**2)
					# plt.subplot(1,3,1)
					# plt.imshow(cropped_im[-1].numpy().T)
					# plt.subplot(1,3,2)
					# plt.imshow(mask.T)
					# plt.subplot(1,3,3)
					# plt.imshow(image.numpy().T)
					# plt.title("Tampered Cropped:{} Y:{} Percent:{}".format(cropped_im[-1].shape,Y[i],np.sum(mask[y:y+crop_size,x:x+crop_size])/crop_size**2))
					# plt.show()
		try:			
			im = torch.stack(cropped_im,0)
		except Exception  as e:
			print("Error:",e)
			print("Not able to get any tampered image")
			print(Y)
			os._exit(0)
		# Random permute for training
		# rand_ind = np.random.permutation(np.arange(len(cropped_y)))
		# im = im[rand_ind,...]		
		cropped_y = np.array(cropped_y)
		# cropped_y = cropped_y[rand_ind]		
	else:	
		for i,image in enumerate(im):
			if Y[i] == 1:
				mask,(y1,y2),(x1,x2) = mask_list[i]
			else:
				mask = None
			print
			patches,y = get_patches(image,mask)
			cropped_im = cropped_im + list(patches)
			cropped_y = cropped_y + y
		cropped_im = np.array(cropped_im)			
		im = torch.Tensor(cropped_im)


	y = torch.LongTensor(cropped_y)


	# print(im.shape[0],len(percent_list))
	# for ind in range(im.shape[0]):
	# 	plt.imshow(im[ind,...].numpy().T)
	# 	plt.title("{} {}".format(y[ind],percent_list[ind]))
	# 	plt.show()


	return im,y,percent_list


def create_samplers(length,split):
	"""
		To make a train and validation split 
		we must know out of which indices should
		the dataloader load for training images and validation images
	"""

	# Validation dataset size
	val_max_size = np.floor(length*split).astype('int')

	# List of Randomly sorted indices
	idx = np.arange(length)
	idx = np.random.permutation(idx)

	# Make a split
	train_idx = idx[0:val_max_size]
	validation_idx = idx[val_max_size:length]

	# Create the sampler required by dataloaders
	train_sampler = SubsetRandomSampler(train_idx)
	val_sampler = SubsetRandomSampler(validation_idx)

	return train_sampler,val_sampler


if __name__ == "__main__":
	pass
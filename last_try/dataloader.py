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
	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class DataSet(torch.utils.data.Dataset):
	"""
		Our dataset loader for each training, testing, val dataset

	"""
	def __init__(self,opt, fake_images):
		super().__init__()
		# Directories
		self.fake_dir = fake_images
		# Calculate filename list 
		fake_list = [ filename for filename in os.listdir(self.fake_dir)]		
		self.fake_list = fake_list
		self.opt = opt
		print("Dataset size Images(0):{}".format(len(self.fake_list)))
	
	def __getitem__(self,idx):
	
		"""
			Given an index it returns the image and its ground truth (fake: 1, true: 0)
		"""
		try:
			path = os.path.join(self.fake_dir, self.fake_list[idx])
			im = Image.open(path).convert('RGB')
		except Exception as e:
			idx += 1
			path = os.path.join(self.fake_dir, self.fake_list[idx])
			im = Image.open(path).convert('RGB')

		im = train_transforms(im)
		_,fileno, patch_y,patch_x,total_x,total_y,y = self.fake_list[idx].split('.')[0].split('_') 
		fileno = int(fileno)
		patch_y = int(patch_y)
		patch_x = int(patch_x)
		total_y = int(total_y)
		total_x = int(total_x)

		y = int(y)
		y =  torch.LongTensor([y])

		return im,y,fileno,patch_y,patch_x,total_y,total_x				

	def __len__(self):
		return len(self.fake_list)


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
import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class DataSet(torch.utils.data.Dataset):
	"""
		Our dataset loader for each training, testing, val dataset

	"""
	def __init__(self,opt,true_images, fake_images):
		super().__init__()
		# Directories
		self.true_dir = true_images
		self.fake_dir = fake_images
		
		# Calculate filename list 
		self.true_list = [ filename for filename in os.listdir(self.true_dir)]		
		self.fake_list = [ filename for filename in os.listdir(self.fake_dir)]		

		self.opt = opt

	def __getitem__(self,idx):

		"""
			Given an index it returns the image and its ground truth (fake: 1, true: 0)
		"""
		# idx = idx % 100 
		# if np.random.random() > 0.5:
		# 	idx = idx +  len(self.true_list)
		if idx >= len(self.true_list):
			path = os.path.join(self.fake_dir, self.fake_list[idx - len(self.true_list)])
			y = 1
		else:
			path = os.path.join(self.true_dir, self.true_list[idx])
			y = 0

		# Load image as rgb
		im = Image.open(path).convert('RGB').resize((self.opt.load_size,self.opt.load_size))

		im = transforms.ToTensor()(im)
		return im,y

	def __len__(self):
		return len(self.true_list) + len(self.fake_list)

def collate_fn(data):
	"""
	Creates mini-batch tensors from the list of tuples (image, caption).
	We will have multiple images of people for each datapoint
	Args:
		data: list of tuple (image, persons). 
			- image: torch tensor of shape (3, 256, 256).
			- persons: persons detected in the image.
	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		person_images: torch tensor of total persons present over all images
		target_ind = image_index of original for each person_image's output
	"""
	# Sort a data list by caption length (descending order).
	im, y = zip(*data)
	# Merge images (from tuple of 3D tensor to 4D tensor).
	im = torch.stack(im, 0)
	y = torch.LongTensor(y)

	return im,y


def create_samplers(length,split):
	val_max_size = np.floor(length*split).astype('uint8')
	idx = range(length)
	validation_idx = np.random.choice(idx, size=val_max_size, replace=False)
	train_idx = list( set(idx)  - set(validation_idx))

	train_sampler = SubsetRandomSampler(train_idx)
	val_sampler = SubsetRandomSampler(validation_idx)

	return train_sampler,val_sampler


if __name__ == "__main__":
	pass
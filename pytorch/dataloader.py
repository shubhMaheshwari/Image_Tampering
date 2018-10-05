import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io
import torch
from torchvision.transforms import transforms
from options import TrainOptions,TestOptions


class DataSet(torch.utils.data.Dataset):
	"""
		Our dataset loader for each training, testing, val dataset

	"""
	def __init__(self,opt):
		super().__init__()
		self.sample_dir = opt.sample_dir
		self.target_dir = opt.target_dir
		
		# Calculate dir images 
		self.sample_list = []		
		self.target_list = []		
		self.opt = opt


	def __getitem__(self,idx):
		data_point = self.data[idx]

		# Get Image path
		filename = data_point[0][0]
		dir_name = data_point[1][0]
		path = os.path.join(dir_name,filename)
		path = os.path.join(self.dataset_path,path)
		
		# Load image as rgb
		im = Image.open(path).convert('RGB')

		# Offset for not training set
		r = 2 if self.data_type != "train" else 0
		# Get all the informations about users 
		# and store them in a dict 
		persons = []
		for elem in data_point[4][0]:
			details = {}  
			box = elem[0][0].astype(int)
			details['image'] = im.crop((box[0],box[1],box[2],box[3])).resize((self.opt.load_size,self.opt.load_size))
			details['image'] = emotic_transform(details['image'])
			details['box'] = elem[0][0]
			details['emotions'] = [ emotion_dict[e[0]] -1 for e in elem[1][0][0][0][0] ] 
			details['VAD'] = [ e[0][0]/10 for e in elem[r + 2][0][0] ]
			details['gender'] = gender_dict[elem[r + 3][0]] -1
			details['age'] = age_dict[elem[r + 4][0]] -1
			
			persons.append(details)

		im = im.resize((self.opt.load_size,self.opt.load_size))
		im = emotic_transform(im)
		return (im,persons)

	def __len__(self):
		return self.data.shape[0]

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
	im, multi_persons = zip(*data)
	# Merge images (from tuple of 3D tensor to 4D tensor).
	im = torch.stack(im, 0)
	target_ind = []
	person_images = []
	person_emotions = []
	person_VAD = []
	person_gender = []
	person_age = []
	# Take all the person images and labels to make a single 4D tensor

	for i,persons in enumerate(multi_persons):
		for j,person in enumerate(persons):
			target_ind.append(i)
			person_images.append(person['image'])

			person_emotion_list  = torch.zeros((26,))
			person_emotion_list[person['emotions']]  = 1
			person_emotions.append(person_emotion_list)
			person_VAD.append(person['VAD'])
			person_age.append(person['age'])
			person_gender.append([person['gender']])

	person_images = torch.stack(person_images,0)
	true_emotions = torch.stack(person_emotions,0)
	target_ind = torch.LongTensor(target_ind)
	true_VAD = torch.FloatTensor(person_VAD)
	true_age = torch.LongTensor(person_age)
	true_gender = torch.FloatTensor(person_gender)

	return im,person_images,target_ind,(true_emotions,true_VAD,true_age,true_gender)

if __name__ == "__main__":
	train_opt = TrainOptions().parse()
	train_data = DataSet('train',train_opt)
	train_loader = torch.utils.data.DataLoader(train_data,collate_fn=collate_fn,
					batch_size=train_opt.batch_size,shuffle=True,num_workers=2)
	test_opt = TestOptions().parse()	
	test_loader =  DataSet('test',test_opt)
	val_loader =   DataSet('val',test_opt)

	
	for im,person_images,target_ind,labels in train_loader:
		print(labels)

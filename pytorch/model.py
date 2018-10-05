import torch 
import torch.nn as nn


class Model(nn.Module):
	"""
		Our model 
		It is normal CNN, we will keep on updating it until 
		we get the desired result 

		Trial 1 =>  VGG network. 
	"""
	def __init__(self,opt):
		super(Model,self).__init__()
		
		self.conv_list = []
		last_size = 3
		for i in range(opt.num_layers):
			if last_size == 3:
				last_size = 16
				conv_layer1 = nn.Conv2d(3,last_size,kernel_size=5,padding=2)
			else:
				last_size = min(256,last_size*2)
				conv_layer1 = nn.Conv2d(last_size//2,last_size,kernel_size=5,padding=2)

			self.conv_list.append(conv_layer1)
			conv_layer2 = nn.Conv2d(last_size,last_size,kernel_size=5,padding=2)
			self.conv_list.append(conv_layer2)
			
			# Instead of pooling strided conv
			conv_layer3 = nn.Conv2d(last_size,last_size,kernel_size=5,stride=opt.stride,padding=2)
			self.conv_list.append(conv_layer3)
	
		self.conv_list = nn.ModuleList(self.conv_list)

		self.fc0 = nn.Linear(128,64)

		self.fcf = nn.Linear(64,2)
		self.fct = nn.Linear(64,2)


	def conv_forward(self,images):
		"""
			Runs all our convolution filters on images and result filters
			:param images: 4D torch tensor 
			return: Nx16x16 torch tensor, features of images
		"""
		for conv_layer in self.conv_list:
			images = conv_layer(images)

		images = images.view(images.shape[0],-1)	

		images = torch.relu(self.fc0(images))
		return images
		

	def forward(self,main_images,tampered_image):
		"""
			Compute the predictions for both images and also calculate the MMD
			@param main_images: batch of sample images
			@param tampered_image: batch of fake images
			@return tuple(
					prediction for sample images, 
					prediction for fake images,
					MMD loss for improving feature embeddings
						)
 		"""

		# Get features from images
		main_features = self.conv_forward(main_images)
		tmain_features = self.conv_forward(tampered_image)
		
		# Run an MLP on sample images for getting predictions
		features1 = torch.relu(self.fct(main_features))

		# Run an MLP on sample images for getting predictions
		tfeatures1 = torch.relu(self.fct(tmain_features))



		return features1,tfeatures1,torch.mean(main_features - tmain_features)



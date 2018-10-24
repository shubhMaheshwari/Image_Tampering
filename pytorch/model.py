import torch 
import torch.nn as nn


class Model(nn.Module):
	"""
		Our model 
		It is normal CNN, we will keep on updating it until 
		we get the desired result 

		Trial 1 =>  8 layer CNN network. 
	"""
	def __init__(self,opt):
		super(Model,self).__init__()
		self.opt = opt


		self.conv_list = []	
		self.conv_list.append(nn.Conv2d(3,32,kernel_size=5,stride=2,padding=2))
		self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
		self.conv_list.append(nn.BatchNorm2d(32))
		self.conv_list.append(nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1))
		self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
		self.conv_list.append(nn.BatchNorm2d(64))
		self.conv_list.append(nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1))
		self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
		self.conv_list.append(nn.BatchNorm2d(128))
		self.conv_list.append(nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1))
		self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
		self.conv_list.append(nn.BatchNorm2d(128))
		self.conv_list.append(nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1))
		self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
		self.conv_list.append(nn.BatchNorm2d(128))
	
		self.conv_list = nn.ModuleList(self.conv_list)

		self.fc0 = nn.Linear(512,256)
		self.r0 = nn.LeakyReLU(0.2, inplace=True)

		self.d0 = nn.Dropout(p=0.5)		

		self.fcs1 = nn.Linear(256,32)
		self.sr1 = nn.LeakyReLU(0.2, inplace=True)
		self.sd1 = nn.Dropout(p=0.5)
		self.fcs2 = nn.Linear(32,2)

		self.fct1 = nn.Linear(256,32)
		self.st1 = nn.LeakyReLU(0.2, inplace=True)
		self.td1 = nn.Dropout(p=0.5)
		self.fct2 = nn.Linear(32,2)

		self.mmd = MMD_LOSS(opt)


	def conv_forward(self,images):
		"""
			Runs all our convolution filters on images and result filters
			:param images: 4D torch tensor 
			return: Nx16x16 torch tensor, features of images
		"""
		for conv_layer in self.conv_list:
			images = conv_layer(images)


		images = images.view(images.shape[0],-1)	
		images = self.r0(self.fc0(images))
		return images
		

	def forward(self,main_images,target_image):
		"""
			Compute the predictions for both images and also calculate the MMD
			@param main_images: batch of sample images
			@param target_image: batch of target images
			@return tuple(
					prediction for sample images, 
					prediction for fake images,
					MMD loss for improving feature embeddings
						)
 		"""



		# Get features from images
		main_features = self.d0(self.conv_forward(main_images))
		tmain_features = self.d0(self.conv_forward(target_image))
		
		# Run an MLP on sample images for getting predictions
		features = self.sd1(self.sd1(self.fcs1(main_features)))
		pred_sample = torch.softmax(self.fcs2(features),dim=1)

		# Run an MLP on target images for getting predictions
		tfeatures = self.td1(self.td1(self.fcs1(tmain_features)))
		pred_target = torch.softmax(self.fcs2(tfeatures),dim=1)

		# print(pred_target)
		# print(pred_sample)


		mmd = self.mmd(features,tfeatures)
		return pred_sample,pred_target,mmd


class MMD_LOSS():
	"""
		MMD LOSS using Gausian Kernel to calculate simlilarity between 2 vector spaces 
	"""
	def __init__(self,opt):
		super(MMD_LOSS,self).__init__()

		# Denotes the possible widths 
		self.sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
		self.sigmas = torch.Tensor(self.sigmas).to(opt.device)
	def _gaussian_kernel(self,x,y):

		beta = 1./(2.*self.sigmas)
		beta = beta.reshape((1,-1)).t()
		pair_wise_distance = torch.mean(x.unsqueeze(0) - y.unsqueeze(1),2)**2
		
		s = torch.mm(beta,pair_wise_distance.reshape(1,-1))
		g = torch.sum(torch.exp(-s),0).reshape(pair_wise_distance.shape)
		# torch.exp(-0.5*cd ((x - y)**2)/)
		return g
	def __call__(self, x,y):
		# \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
		cost = torch.mean(self._gaussian_kernel(x,x)) + torch.mean(self._gaussian_kernel(y,y)) -2*torch.mean(self._gaussian_kernel(x,y)) 

		return cost
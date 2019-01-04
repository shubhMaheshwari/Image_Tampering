import torch 
import torch.nn as nn
import torchvision.models as models



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

		self.conv_list = nn.Sequential(
			nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.BatchNorm2d(32),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.BatchNorm2d(64),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.BatchNorm2d(128),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.BatchNorm2d(128),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.BatchNorm2d(128),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.BatchNorm2d(128),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size=2,stride=2),
			)

		self.sample_features = nn.Linear(512,256)		
		self.sample = nn.Sequential(
			# nn.Dropout(p=0.5),
			nn.LeakyReLU(0.2, inplace=True),			
			nn.Dropout(p=0.1),
			nn.Linear(256,2),
			# nn.LeakyReLU(0.2, inplace=True),
			# nn.Linear(32,2)
		)

		self.target_features = nn.Linear(512,256)				
		self.target = nn.Sequential(
			# nn.Dropout(p=0.5),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(256,2),
			# nn.LeakyReLU(0.2, inplace=True),
			# nn.Linear(32,2)
		)

		self.mmd = MMD_LOSS(opt)

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
		main_features = self.conv_list(main_images)
		main_features = main_features.view(main_features.size(0),512)
		main_features = self.sample_features(main_features)


		tmain_features = self.conv_list(target_image)
		tmain_features = tmain_features.view(tmain_features.size(0),512)
		tmain_features = self.target_features(tmain_features)

		
		# Run an MLP on sample images for getting predictions
		features = self.sample(main_features)
		pred_sample = torch.softmax(features,dim=1)

		# Run an MLP on target images for getting predictions
		tfeatures = self.target(tmain_features)
		pred_target = torch.softmax(tfeatures,dim=1)

		mmd = self.mmd(main_features,tmain_features)
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
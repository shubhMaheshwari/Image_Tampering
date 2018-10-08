import argparse
import os
import torch 


class BaseOptions(object):
	"""
		Base options given to run the network
	"""
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.device = None
	def initialize(self):
		self.parser.add_argument('--batch_size',type=int, default=128, help='Batch Size for each iternations')
		self.parser.add_argument('--val_batch_size',type=int, default=180, help='Batch Size for each iternations')
		self.parser.add_argument('--load_size',type=int, default=128, help='WxH of the image to be loaded, both original and each persons image')
		self.parser.add_argument('--num_layers',type=int, default=4, help='Number of layers in CNN')
		self.parser.add_argument('--stride',type=int, default=10, help='Stride during convolution')
		self.parser.add_argument('--gpus', default='0', help='-1: cpu else is a list of gpu ids eg. 0,1,2')
		self.parser.add_argument('--use_gpu',type=bool, default=True, help='Whether to use gpu')
		self.parser.add_argument('--sample_true_dir',type=str, default="../coco/images/new_train2014/" ,help='Not tampered images from coco')
		self.parser.add_argument('--sample_fake_dir',type=str, default="../coco/images/CVInpainting/" ,help='tampered images from coco')
		self.parser.add_argument('--target_true_dir',type=str, default="../CASIA/CASIA2/Au" ,help='Not tampered images from CASIA')
		self.parser.add_argument('--target_fake_dir',type=str, default="../CASIA/CASIA2/Tp" ,help='tampered images from CASIA')


	def parse(self):
		"""
			Load the arguments given and processes them for further use
		"""
		self.initialize()
		opt = self.parser.parse_args()
		args = vars(opt)


		if opt.use_gpu:
			gpus = [int(i) for i in opt.gpus.split(',')]
			print("=> active GPUs: {}".format(gpus))
			opt.gpus = gpus

		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		return opt


class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--use_case', type=str, default="Train", help='Why are we running the model eg. Train , test, finetune etc.')
		self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
		self.parser.add_argument('--weight_decay', type=float, default=0.005, help='Regularization')
		self.parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
		self.parser.add_argument('--epoch', type=int, default=200, help='Number of epoch')
		self.parser.add_argument('--lambda_sample', type=float, default=0.5, help='Loss effected by true images')
		self.parser.add_argument('--lambda_target', type=float, default=2.0, help='Loss effected by fake images')
		self.parser.add_argument('--lambda_mmd', type=float, default=100, help='MMD Loss percentage')
		self.parser.add_argument('--print_iter', type=int, default=5, help='Invervals to print between iters')
		self.parser.add_argument('--iter', type=int, default=20, help='Number of interations per epoch')
		self.parser.add_argument('--split_ratio', type=float, default=0.8, help='Number of interations per epoch')
		

		# Visdom settings during training
		self.parser.add_argument('--display_server', type=str, default='http://localhost', help='visdom display host')
		self.parser.add_argument('--display_port', type=int, default=8097, help='Visdom display port')



class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)	
		self.parser.add_argument('--test_cases', type=int, default=128, help='Number of images to run on test case')
		self.parser.add_argument('--epoch', type=int, default=0, help='From check points dir get the epoch to run the model')

if __name__ == "__main__":
	train_opt = TrainOptions().parse()
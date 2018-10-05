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
		self.parser.add_argument('--batch_size',type=int, default=10	, help='Batch Size for each iternations')
		self.parser.add_argument('--load_size',type=int, default=256, help='WxH of the image to be loaded, both original and each persons image')
		self.parser.add_argument('--num_layers',type=int, default=4, help='Number of layers in CNN')
		self.parser.add_argument('--stride',type=int, default=4, help='Stride during convolution')
		self.parser.add_argument('--gpus', default='0', help='-1: cpu else is a list of gpu ids eg. 0,1,2')
		self.parser.add_argument('--use_gpu',type=bool, default=True, help='Whether to use gpu')
		self.parser.add_argument('--sample_dir',type=str, required=True, help='where images from coco or less tampered data is kept')
		self.parser.add_argument('--target_dir',type=str, required=True, help='where images from CASIA, IEEE are stored')

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
		self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
		self.parser.add_argument('--weight_decay', type=float, default=0.0005, help='Regularization')
		self.parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
		self.parser.add_argument('--epoch', type=int, default=200, help='Number of epoch')
		self.parser.add_argument('--lambda_true', type=float, default=0.25, help='Loss effected by true images')
		self.parser.add_argument('--lambda_fake', type=float, default=0.25, help='Loss effected by fake images')
		self.parser.add_argument('--lambda_mmd', type=float, default=2.0, help='MMD Loss percentage')
		self.parser.add_argument('--print_iter', type=int, default=5, help='Invervals to print between iters')
		self.parser.add_argument('--iter', type=int, default=1000, help='Number of interations per epoch')


class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)	
		self.parser.add_argument('--test_cases', type=int, default=128, help='Number of images to run on test case')

if __name__ == "__main__":
	train_opt = TrainOptions().parse()
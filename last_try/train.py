# Main module to train the model, load the data,
# do gradient descent etc. followed by saving our model for later testing
from dataloader import DataSet,create_samplers
from model import Model
from options import TrainOptions
import torch
from torchvision.transforms import *
import torch.optim as optim
import numpy as np 
from visualizer import Visualizer
import os
# Get the Hyperparaeters 
opt = TrainOptions().parse()

import matplotlib.pyplot as plt

sample_dataset = DataSet(opt,"./CASIA_train_patches/")

train_sampler,val_sampler = create_samplers(sample_dataset.__len__(),opt.split_ratio)
sample_loader = torch.utils.data.DataLoader(sample_dataset,sampler=train_sampler,batch_size=opt.batch_size,num_workers=15)
sample_val_loader = torch.utils.data.DataLoader(sample_dataset,sampler=val_sampler,batch_size=opt.val_batch_size,num_workers=5,shuffle=False)

target_dataset = DataSet(opt,"./total_train_patches/")

train_sampler,val_sampler = create_samplers(target_dataset.__len__(),opt.split_ratio)
target_loader = torch.utils.data.DataLoader(target_dataset,sampler=train_sampler,batch_size=opt.batch_size,num_workers=20)
target_val_loader = torch.utils.data.DataLoader(target_dataset,sampler=val_sampler,batch_size=opt.val_batch_size,num_workers=10,shuffle=False)


# Check if gpu available or not
device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")
opt.device = device

# Load the model and send it to gpu
model = Model(opt)
model = model.to(device)
if opt.use_gpu:	
	model = torch.nn.DataParallel(model, device_ids=opt.gpus)

# Print our model 
print('------------ Model -------------')
print(model)
print('-------------- End ----------------')	


# Make checkpoint dir to save best models
if not os.path.exists('./checkpoints'):
	os.mkdir('./checkpoints')

# If require load old weights
if opt.load_epoch > 0:
	model.load_state_dict(torch.load('./checkpoints/' + 'model_{}.pt'.format(opt.load_epoch)))

else:
	def init_weights(m):
		if type(m) == torch.nn.Linear:
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.01)
		elif isinstance(m,torch.nn.Conv2d):
			torch.nn.init.xavier_uniform_(m.weight,gain=np.sqrt(2))
	model.apply(init_weights)
	
# Visualizer using visdom
vis = Visualizer(opt)

# Loss functons(Cross Entropy)
# Adam optimizer
criterion_sample = torch.nn.CrossEntropyLoss().to(device)
criterion_target = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.weight_decay)


# Variables to store the losses and accuracy in list so that we can plot it later
best_epoch = 0
target_acc_list = []
sample_acc_list = []

def save_model(model,epoch):
	filename = './checkpoints/' + 'model_{}.pt'.format(epoch)
	torch.save(model.state_dict(), filename)

def get_accuracy(pred_sample, sample_labels, pred_target, target_labels):
	pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )
	# Getting accuracy on sample images
	sample_acc = np.mean(pred_sample == sample_labels.cpu().data.numpy())
	# Similarly for target images
	pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)
	target_acc = np.mean(pred_target == target_labels.data.numpy())

	return sample_acc, target_acc

# Training loop
for epoch in range(opt.epoch):
	# In each epoch first trained on images and then perform validation

	model.train()
	for i, (sample_images, sample_labels,fileno,patch_y,patch_x,total_y,total_x) in enumerate(sample_loader):			

		sample_labels = sample_labels.squeeze(1)
		
		# target_images = sample_images
		# target_labels = sample_labels
		target_images,target_labels,_,_,_,_,_ =  next(iter(target_loader))
		target_labels = target_labels.squeeze(1)

		# Do a prediction
		try:
			optimizer.zero_grad()
			pred_sample,pred_target,loss_mmd = model(sample_images.to(device),target_images.to(device))
			print(pred_sample.shape,sample_labels.shape,pred_target.shape,target_labels.shape)

			# Calculate loss
			loss_sample = criterion_sample(pred_sample,sample_labels.to(device))
			loss_target = criterion_target(pred_target,target_labels.to(device))


		except RuntimeError  as r:
			print("Error:",r)
			torch.cuda.empty_cache()
			continue
		except Exception as e:
			print("Error:",e)	
			torch.cuda.empty_cache()
			continue

		# Combine loss
		# loss =  opt.lambda_sample*loss_sample + opt.lambda_target*loss_target
		loss =  opt.lambda_sample*loss_sample + opt.lambda_target*loss_target + opt.lambda_mmd*loss_mmd

		# Do backpropogation followed by a gradient descent step
		loss.backward()
		optimizer.step()	

		# # Once in a while print losses and accuracy
		if i % opt.print_iter == 0:
			# As we are using softmax, class with highest probality is predicted
			sample_acc, target_acc = get_accuracy(pred_sample,sample_labels,pred_target,target_labels)

			# Print loss
			print("Iter:{}/{} Loss:{} Sample_Acc:{} Target_Acc:{}".format(i,40, loss.cpu().data.numpy(),sample_acc,target_acc))

			vis.append_metrics(loss, loss_sample, loss_target,loss_mmd, target_acc, sample_acc)

			# Using visdom to visualize the model
			vis.plot_loss()
			vis.plot_acc()

			# Also show images with their prediction and ground truth
			for j in range(min(4,opt.batch_size)):
				vis.show_image(sample_images.cpu().data.numpy()[j,:,:,:],pred_sample[j].cpu().data.numpy(),sample_labels.cpu().data.numpy()[j],display_id=j + 10 ,title="Sample Dataset")
				vis.show_image(target_images.cpu().data.numpy()[j,:,:,:],pred_target[j].cpu().data.numpy(),target_labels.cpu().data.numpy()[j],display_id=j + 15 ,title="Target Dataset")


	# Validate model using the validation set
	model.eval()

	sample_images, sample_labels,fileno,patch_y,patch_x,total_y,total_x = next(iter(sample_val_loader))

	sample_labels = sample_labels.squeeze(1)	
	# target_images = sample_images
	# target_labels = sample_labels
	target_images,target_labels,_,_,_,_,_ =  next(iter(target_loader))
	target_labels = target_labels.squeeze(1)

	try:
		pred_sample,pred_target,loss_mmd = model(sample_images.to(device),target_images.to(device))	
	except RuntimeError  as r:
		torch.cuda.empty_cache()
		continue

	sample_acc, target_acc = get_accuracy(pred_sample,sample_labels,pred_target,target_labels )

	print("Validation:{}th epoch Sample_Acc:{} Target_Acc:{}".format(epoch,sample_acc, target_acc))

	target_acc_list.append(target_acc)
	sample_acc_list.append(sample_acc)

	vis.plot_graph(None,[target_acc_list,sample_acc_list],labels=["Target ACC","Sample ACC"],axis=['Epoch','Acc'] ,display_id=3,title='validation accuracy')

	try:
		if target_acc >= target_acc_list[best_epoch] or epoch%10 ==0:
			save_model(model,epoch)
			best_epoch = epoch
	except Exception as e:
		best_epoch = np.argmax(np.array(target_acc_list))
		print("Error:",e)
		continue
	# Update lr 

	if epoch > opt.lr_decay_iter:
		for g in optimizer.param_groups:
			g['lr'] = opt.lr_decay_param*g['lr']


print("Finished Training, best epoch:",best_epoch)

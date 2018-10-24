# Main module to train the model, load the data,
# do gradient descent etc. followed by saving our model for later testing
from dataloader import DataSet,collate_fn,create_samplers
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



# Load the sample dataset(COCO)
sample_dataset = DataSet(opt,opt.sample_true_dir,opt.sample_fake_dir,opt.sample_fake_dir_mask,opt.sample_dataset)

# Do a dataset split for validation and train
train_sample_sampler,val_sample_sampler = create_samplers(sample_dataset.__len__(),opt.split_ratio)

sample_loader = torch.utils.data.DataLoader(sample_dataset,collate_fn= lambda x: collate_fn(x, opt.num_crops),sampler=train_sample_sampler,
				batch_size=opt.batch_size,num_workers=5)

sample_val_loader = torch.utils.data.DataLoader(sample_dataset,collate_fn=lambda x: collate_fn(x, -1),sampler=val_sample_sampler,
				batch_size=1,num_workers=5)

# Same method for target dataset(CASIA V2)
target_dataset = DataSet(opt,opt.target_true_dir,opt.target_fake_dir,opt.target_fake_dir_mask,opt.target_dataset)
train_target_sampler,val_target_sampler = create_samplers(target_dataset.__len__(),opt.split_ratio)
target_loader = torch.utils.data.DataLoader(target_dataset,collate_fn=lambda x: collate_fn(x,  opt.num_crops),sampler=train_target_sampler,
				batch_size=opt.batch_size,num_workers=2)
target_val_loader = torch.utils.data.DataLoader(target_dataset,collate_fn=lambda x: collate_fn(x, -1),sampler=val_target_sampler,
				batch_size=1,num_workers=2)

# Check if gpu available or not
device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")
opt.device = device

# Load the model and send it to gpu
model = Model(opt)
if opt.use_gpu:
	model = model.to(device)	
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

# Visualizer using visdom
vis = Visualizer(opt)

# Loss functons(Cross Entropy)
# Adam optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.weight_decay)


# Variables to store the losses and accuracy in list so that we can plot it later
best_epoch = 0
loss_list = []
loss_sample_list = []
loss_target_list = []
loss_mmd_list = []
target_acc_list = []
train_target_acc_list = []
sample_acc_list = []
train_sample_acc_list = []


def save_model(model,epoch):
	filename = './checkpoints/' + 'model_{}.pt'.format(epoch)
	torch.save(model.state_dict(), filename)

# If K patches are detected tampered then the image is tampered(Only during testing and validating)
K = np.arange((opt.load_size//opt.crop_size)**2)


# Training loop
for epoch in range(opt.epoch):
	model.train()
	# In each epoch first trained on images and then perform validation
	for i in range(opt.iter):

		# Load images
		sample_images, sample_labels = next(iter(sample_loader))
		target_images, target_labels = next(iter(target_loader))

		# print(sample_labels)
		# Do a prediction
		pred_sample,pred_target,loss_mmd = model(sample_images.to(device),target_images.to(device))

		# Calculate loss
		loss_sample = criterion(pred_sample,sample_labels.to(device))
		loss_target = criterion(pred_target,target_labels.to(device))

		# Combine loss
		loss = opt.lambda_sample*loss_sample + opt.lambda_target*loss_target + opt.lambda_mmd*loss_mmd

		# Do backpropogation followed by a gradient descent step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()	

		# Once in a while print losses and accuracy
		if i % opt.print_iter == 0:

			# As we are using softmax, class with highest probality is predicted
			pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )

			# Getting accuracy on sample images
			sample_acc = np.mean(pred_sample == sample_labels.cpu().data.numpy())

			# Similarly for target images
			pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)
			target_acc = np.mean(pred_target == target_labels.data.numpy())

			# Print loss
			print("Iter:{}/{} Loss:{} MMD_Loss:{} Sample_Acc:{} Target_Acc:{}".format(i,opt.iter, loss.data[0],loss_mmd.data[0],sample_acc,target_acc))


			# Upate the loss list and plot it
			loss_list.append(loss.data)
			loss_sample_list.append(loss_sample.data)
			loss_target_list.append(loss_target.data)
			loss_mmd_list.append(loss_mmd.data)

			train_target_acc_list.append(target_acc)
			train_sample_acc_list.append(sample_acc)

			# Using visdom to visualize the model
			vis.plot_graph(None,[loss_list,loss_sample_list,loss_target_list,loss_mmd_list],["Loss","Sample Loss", "Target Loss", "Mmd Loss"] ,display_id=1,title=' loss over time',axis=['Epoch','Loss'])
			vis.plot_graph(None,[train_target_acc_list,train_sample_acc_list],["Train Target ACC","Train Sample ACC"] ,display_id=4,title='Training Accuracy',axis=['Epoch','Acc'])

			# Also show images with their prediction and ground truth
			vis.show_image(sample_images.cpu().data.numpy()[0,:,:,:],pred_sample[0],sample_labels.cpu().data.numpy()[0],display_id=2,title="Sample Dataset")
			# vis.show_image(target_images.cpu().data.numpy()[0,:,:,:],pred_target[0],target_labels.cpu().data.numpy()[0],display_id=5,title= "Target Dataset")


			# print("Sample dataset:")
			# print(pred_sample)
			# print(sample_labels)
			# print("Target Dataset:")
			# print(pred_target)
			# print(target_labels)

	# Validate model using the validation set
	model.eval()

	sample_val_list = []
	pred_sample_val_list = []
	target_val_list = []
	pred_target_val_list = []
	for i in range(opt.val_batch_size):
		sample_images, sample_labels = next(iter(sample_val_loader))
		target_images, target_labels = next(iter(target_val_loader))

		pred_sample,pred_target,loss_mmd = model(sample_images.to(device),target_images.to(device))	
		pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )
		pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)

		print(pred_target)

		sample_labels = np.sum(sample_labels.numpy(),0) 
		pred_sample = np.sum(pred_sample,0)
		# print("Misclassied {} from Sample Image ".format(abs(pred_sample-sample_labels)))

		target_labels = np.sum(target_labels.numpy(),0)
		pred_target = np.sum(pred_target,0)
		# print("Misclassied {} from Target Image ".format(abs(pred_target-target_labels)))

		sample_val_list.append(np.sign(sample_labels))
		pred_sample_val_list.append( (( pred_sample - K) > 0).astype('int'))
		target_val_list.append(np.sign(target_labels - K))
		pred_target_val_list.append(( (K - pred_target) > 0).astype('int'))

	sample_val_list = np.array(sample_val_list)
	pred_sample_val_list = np.array(pred_sample_val_list).T

	sample_acc = np.mean(sample_val_list == pred_sample_val_list,axis=1)

	target_val_list = np.array(target_val_list)
	pred_target_val_list = np.array(pred_target_val_list).T

	target_acc = np.mean(target_val_list == pred_target_val_list,axis=1)


	best_sample_K = np.argmax(sample_acc)
	best_sample_acc = target_acc[best_sample_K]

	best_target_K = np.argmax(target_acc)
	best_target_acc = target_acc[best_target_K]

	print("Validation:{}th epoch Sample_Acc:{} Target_Acc:{} Best_K(Sample):{} Best_K(target) :{}".format(epoch,best_sample_acc,best_target_acc,best_sample_K,best_target_K))


	if epoch == 0:
		save_model(model,epoch)
		best_epoch = epoch
	elif best_target_acc >= target_acc_list[best_epoch]:
		save_model(model,epoch)
		best_epoch = epoch

	target_acc_list.append(best_target_acc)
	sample_acc_list.append(best_sample_acc)

	vis.plot_graph(None,[target_acc_list,sample_acc_list],labels=["Target ACC","Sample ACC"],axis=['Epoch','Acc'] ,display_id=3,title='validation accuracy')


	# # Update lr 
	if epoch > opt.lr_decay_iter:
		for g in optimizer.param_groups:
			g['lr'] = opt.lr_decay_param*g['lr']



print("Finished Training, best epoch:",best_epoch)
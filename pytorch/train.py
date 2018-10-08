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
sample_dataset = DataSet(opt,opt.sample_true_dir,opt.sample_fake_dir)

# Do a dataset split for validation and train
train_sample_sampler,val_sample_sampler = create_samplers(sample_dataset.__len__(),opt.split_ratio)

sample_loader = torch.utils.data.DataLoader(sample_dataset,collate_fn=collate_fn,sampler=train_sample_sampler,
				batch_size=opt.batch_size,num_workers=2)

sample_val_loader = torch.utils.data.DataLoader(sample_dataset,collate_fn=collate_fn,sampler=val_sample_sampler,
				batch_size=opt.val_batch_size,num_workers=2)

# Same method for target dataset(CASIA V2)
target_dataset = DataSet(opt,opt.target_true_dir,opt.target_fake_dir)
train_target_sampler,val_target_sampler = create_samplers(target_dataset.__len__(),opt.split_ratio)
target_loader = torch.utils.data.DataLoader(target_dataset,collate_fn=collate_fn,sampler=train_target_sampler,
				batch_size=opt.batch_size,num_workers=2)
target_val_loader = torch.utils.data.DataLoader(target_dataset,collate_fn=collate_fn,sampler=val_target_sampler,
				batch_size=opt.val_batch_size,num_workers=2)

# Load the model and send it to gpu
model = Model(opt)
device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")
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



# Training loop
for epoch in range(opt.epoch):
	model.train()

	# In each epoch first trained on images and then perform validation
	for i in range(opt.iter):

		# Load images
		sample_images, sample_labels = next(iter(sample_loader))
		target_images, target_labels = next(iter(target_loader))

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
			vis.plot_graph(None,[loss_list,loss_sample_list,loss_target_list,loss_mmd_list],["Loss","Sample Loss", "Target Loss", "Mmd Loss"] ,display_id=1)
			vis.plot_graph(None,[train_target_acc_list,train_sample_acc_list],["Train Target ACC","Train Sample ACC"] ,display_id=4)

			# Also show images with their prediction and ground truth
			vis.show_image(sample_images.cpu().data.numpy()[0,:,:,:],pred_sample[0],sample_labels.cpu().data.numpy()[0],display_id=2)
			vis.show_image(target_images.cpu().data.numpy()[0,:,:,:],pred_target[0],target_labels.cpu().data.numpy()[0],display_id=5)


	# Validate model using the validation set
	model.eval()
	sample_images, sample_labels = next(iter(sample_val_loader))
	target_images, target_labels = next(iter(target_val_loader))

	pred_sample,pred_target,loss_mmd = model(sample_images.to(device),target_images.to(device))	

	pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )
	sample_acc = np.mean(pred_sample == sample_labels.cpu().data.numpy())

	pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)
	target_acc = np.mean(pred_target == target_labels.data.numpy())

	print("Validation:{}th epoch Sample_Acc:{} Target_Acc:{}".format(epoch,sample_acc,target_acc))


	if epoch == 0:
		save_model(model,epoch)
		best_epoch = epoch
	elif target_acc >= target_acc_list[best_epoch]:
		save_model(model,epoch)
		best_epoch = epoch

	target_acc_list.append(target_acc)
	sample_acc_list.append(sample_acc)

	vis.plot_graph(None,[target_acc_list,sample_acc_list],["Target ACC","Sample ACC"] ,display_id=3)
	print(pred_sample)
	print(sample_labels)


print("Finished Training, best epoch:",best_epoch)
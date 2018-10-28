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

import matplotlib.pyplot as plt

# Load the sample datasets
sample_path_list = [
					("Inpainting","../coco/images/new_train2014/","../coco/images/CVIPmasktrain2014/","../coco/images/Inpaintingtrain2014/"),
					("Segment","../coco/images/new_train2014/","../coco/images/mask2014/","../coco/images/train2014semantics/"),
					# ("Adobe_Featuring","../coco/images/new_train2014/","../featuring_dataset/alpha/","../featuring_dataset/merged/"),
					# ("IEEE","../IEEE/training/phase-01/training/pristine/","../IEEE/training/phase-01/training/masks/","../IEEE/training/phase-01/training/fake/")

					]
target_path_list = [("IEEE","../IEEE/training/phase-01/training/pristine/","../IEEE/training/phase-01/training/masks/","../IEEE/training/phase-01/training/fake/")]

# Load all the sample datasets
sample_loaders =[]
for paths in sample_path_list:
	dataset_name,true_path,mask_path,fake_path = paths
	dataset = DataSet(opt,true_path,fake_path,mask_path,dataset_name)


	# Do a dataset split for validation and train
	train_sampler,val_sampler = create_samplers(dataset.__len__(),opt.split_ratio)

	loader = torch.utils.data.DataLoader(dataset,collate_fn= lambda x: collate_fn(x, opt.num_crops),sampler=train_sampler,
				batch_size=opt.batch_size,num_workers=10)

	val_loader = torch.utils.data.DataLoader(dataset,collate_fn=lambda x: collate_fn(x, -1),sampler=val_sampler,
				batch_size=1,num_workers=10)

	sample_loaders.append((loader,val_loader))

# Load all the target datasets
target_loaders =[]
for paths in target_path_list:
	dataset_name,true_path,mask_path,fake_path = paths
	dataset = DataSet(opt,true_path,fake_path,mask_path,dataset_name)


	# Do a dataset split for validation and train
	train_sampler,val_sampler = create_samplers(dataset.__len__(),opt.split_ratio)

	loader = torch.utils.data.DataLoader(dataset,collate_fn= lambda x: collate_fn(x, opt.num_crops),sampler=train_sampler,
				batch_size=opt.batch_size,num_workers=1)

	val_loader = torch.utils.data.DataLoader(dataset,collate_fn=lambda x: collate_fn(x, -1),sampler=val_sampler,
				batch_size=1,num_workers=1)

	target_loaders.append((loader,val_loader))	



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
loss_list = []
loss_sample_list = []
loss_target_list = []
loss_mmd_list = []
target_acc_list = []
train_target_acc_list = []
sample_acc_list = []
train_sample_acc_list = []
best_K_list = []

def save_model(model,epoch):
	filename = './checkpoints/' + 'model_{}.pt'.format(epoch)
	torch.save(model.state_dict(), filename)

# If K patches are detected tampered then the image is tampered(Only during testing and validating)
K = np.arange((opt.load_size//opt.crop_size)**2)


# Training loop
for epoch in range(opt.epoch):
	# In each epoch first trained on images and then perform validation

	for sample_loader, sample_val_loader in sample_loaders:
		for target_loader, target_val_loader in target_loaders:
			
			model.train()
			for i in range(opt.iter):
				sample_images, sample_labels,spercent_list = next(iter(sample_loader))
				target_images, target_labels,tpercent_list = next(iter(target_loader))
				# Do a prediction
				optimizer.zero_grad()
				try:
					pred_sample,pred_target,loss_mmd = model(sample_images.to(device),target_images.to(device))
				except RuntimeError  as r:
					print("Error:",r)
					torch.cuda.empty_cache()
					continue
				# Calculate loss
				loss_sample = criterion_sample(pred_sample,sample_labels.to(device))
				loss_target = criterion_target(pred_target,target_labels.to(device))

				# Combine loss
				loss =  opt.lambda_sample*loss_sample + opt.lambda_target*loss_target + opt.lambda_mmd*loss_mmd

				# Do backpropogation followed by a gradient descent step
				loss.backward()
				optimizer.step()	

				

				# Once in a while print losses and accuracy
				if i % opt.print_iter == 0:

					print(pred_sample)

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
					for i in range(min(4,opt.batch_size)):
						vis.show_image(sample_images.cpu().data.numpy()[i,:,:,:],pred_sample[i],sample_labels.cpu().data.numpy()[i],display_id=i + 10,title="Sample Dataset:{}".format(spercent_list[i]))
						vis.show_image(target_images.cpu().data.numpy()[i,:,:,:],pred_target[i],target_labels.cpu().data.numpy()[i],display_id=15 + i,title="Target Dataset:{}".format(tpercent_list[i]))


			# Validate model using the validation set
			model.eval()

			sample_val_list = []
			pred_sample_val_list = []
			target_val_list = []
			pred_target_val_list = []
			for i in range(opt.val_batch_size):

				sample_images, sample_labels,_ = next(iter(sample_val_loader))
				target_images, target_labels,_ = next(iter(target_val_loader))

				try:
					pred_sample,pred_target,loss_mmd = model(sample_images.to(device),target_images.to(device))	
					
				except RuntimeError  as r:
					torch.cuda.empty_cache()
					continue
				# print(pred_target)
				# print(sample_images.shape)
				# print(target_images.shape)

				pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )
				pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)

				sample_labels = np.sum(sample_labels.numpy(),0) 
				pred_sample = np.sum(pred_sample,0)
				print("Misclassied {} from Sample Image ".format(abs(pred_sample-sample_labels)))

				target_labels = np.sum(target_labels.numpy(),0)
				pred_target = np.sum(pred_target,0)
				# print(pred_target)
				print("Misclassied {} from Target Image ".format(abs(pred_target-target_labels)))

				sample_val_list.append(np.sign(sample_labels))
				pred_sample_val_list.append( (( pred_sample - K) > 0).astype('int'))
				target_val_list.append(np.sign(target_labels))
				pred_target_val_list.append(( ( pred_target - K) > 0).astype('int'))

			if len(sample_val_list) == 0:
				continue
			
			sample_val_list = np.array(sample_val_list)
			pred_sample_val_list = np.array(pred_sample_val_list).T
	

			sample_acc = np.mean(sample_val_list == pred_sample_val_list,axis=1)

			if len(target_val_list) == 0:
				continue

			target_val_list = np.array(target_val_list)
			pred_target_val_list = np.array(pred_target_val_list).T

			target_acc = np.mean(target_val_list == pred_target_val_list,axis=1)

			best_sample_K = np.argmax(sample_acc)
			best_sample_acc = sample_acc[best_sample_K]

			best_target_K = np.argmax(target_acc)
			best_target_acc = target_acc[best_target_K]

			best_K_list.append(best_target_K)

			print("Validation:{}th epoch Sample_Acc:{} Target_Acc:{} Best_K(Sample):{} Best_K(target) :{}".format(epoch,best_sample_acc,best_target_acc,best_sample_K,best_target_K))

			target_acc_list.append(best_target_acc)
			sample_acc_list.append(best_sample_acc)

			vis.plot_graph(None,[target_acc_list,sample_acc_list],labels=["Target ACC","Sample ACC"],axis=['Epoch','Acc'] ,display_id=3,title='validation accuracy')

	save_model(model,epoch)
	try:
		if epoch == 0:
			best_epoch = epoch
		elif best_target_acc >= target_acc_list[best_epoch]:
			save_model(model,epoch)
			best_epoch = epoch
	except Exception as e:
		best_epoch = np.argmax(np.array(target_acc_list))
		print("Error:",e)
		continue
	# # Update lr 
	if epoch > opt.lr_decay_iter:
		for g in optimizer.param_groups:
			g['lr'] = opt.lr_decay_param*g['lr']

# np.savez("/media/shubh/Pratik (120GB)/Train_file",sample_numpy_images,sample_numpy_y,target_numpy_images,target_numpy_y)

print("Finished Training, best epoch:",best_epoch)
plt.hist(best_K_list)
plt.show()
# Main module to train the model, load the data,
# do gradient descent etc.
from dataloader import DataSet,collate_fn,create_samplers
from model import Model
from options import TrainOptions
import torch
from torchvision.transforms import *
import torch.optim as optim
import numpy as np 
from visualizer import Visualizer
# Get the Hyperparaeters 
opt = TrainOptions().parse()


# Load the dataset
sample_dataset = DataSet(opt,opt.sample_true_dir,opt.sample_fake_dir)

# Do a dataset split for eval and train
train_sample_sampler,val_sample_sampler = create_samplers(sample_dataset.__len__(),opt.split_ratio)

sample_loader = torch.utils.data.DataLoader(sample_dataset,collate_fn=collate_fn,sampler=train_sample_sampler,
				batch_size=opt.batch_size,num_workers=2)

sample_val_loader = torch.utils.data.DataLoader(sample_dataset,collate_fn=collate_fn,sampler=val_sample_sampler,
				batch_size=opt.batch_size*10,num_workers=2)

# Same method for target dataset
target_dataset = DataSet(opt,opt.target_true_dir,opt.target_fake_dir)
train_target_sampler,val_target_sampler = create_samplers(target_dataset.__len__(),opt.split_ratio)
target_loader = torch.utils.data.DataLoader(target_dataset,collate_fn=collate_fn,sampler=train_target_sampler,
				batch_size=opt.batch_size,num_workers=2)
target_val_loader = torch.utils.data.DataLoader(target_dataset,collate_fn=collate_fn,sampler=val_target_sampler,
				batch_size=opt.batch_size*10,num_workers=2)

# Load the model
model = Model(opt)
if opt.use_gpu:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.cuda()	
	model = torch.nn.DataParallel(model, device_ids=opt.gpus)

print('------------ Model -------------')
print(model)
print('-------------- End ----------------')	

# Visualizer 
vis = Visualizer(opt)

# Loss functons
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.weight_decay)

loss_list = []
loss_sample_list = []
loss_target_list = []
loss_mmd_list = []
target_acc_list = []
sample_acc_list = []
# Training loop
for epoch in range(opt.epoch):
	model.train()

	# In each epoch first trained on sample images
	for i in range(opt.iter):


		sample_images, sample_labels = next(iter(sample_loader))
		target_images, target_labels = next(iter(target_loader))

		pred_sample,pred_target,loss_mmd = model(sample_images.cuda(),target_images.cuda())

		loss_sample = criterion(pred_sample,sample_labels.cuda())
		loss_target = criterion(pred_target,target_labels.cuda())

		loss = opt.lambda_sample*loss_sample + opt.lambda_target*loss_target + opt.lambda_mmd*loss_mmd

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()	
		if i % opt.print_iter == 0:

			pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )

			sample_acc = np.mean(pred_sample == sample_labels.cpu().data.numpy())

			pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)
			target_acc = np.mean(pred_target == target_labels.data.numpy())

			print("Iter:{}/{} Loss:{} Sample_Acc:{} Target_Acc:{}".format(i,opt.iter, loss.data[0],sample_acc,target_acc))


			# Upate the loss list and plot it
			loss_list.append(loss.data)
			loss_sample_list.append(loss_sample.data)
			loss_target_list.append(loss_target.data)
			loss_mmd_list.append(loss_mmd.data)

			# Using visdom to visualize the model
			vis.plot_graph(None,[loss_list,loss_sample_list,loss_target_list,loss_mmd_list],["Loss","Sample Loss", "Target Loss", "Mmd Loss"] ,display_id=1)

			vis.show_image(sample_images.cpu().data.numpy()[0,:,:,:],pred_sample[0],sample_labels.cpu().data.numpy()[0],display_id=2)


	model.eval()
	sample_images, sample_labels = next(iter(sample_val_loader))
	target_images, target_labels = next(iter(target_val_loader))
	pred_sample,pred_target,loss_mmd = model(sample_images.cuda(),target_images.cuda())	

	pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )

	sample_acc = np.mean(pred_sample == sample_labels.cpu().data.numpy())

	pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)
	target_acc = np.mean(pred_target == target_labels.data.numpy())

	print("Validation Sample_Acc:{} Target_Acc:{}".format(sample_acc,target_acc))
	target_acc_list.append(target_acc)
	sample_acc_list.append(sample_acc)

	vis.plot_graph(None,[target_acc_list,sample_acc_list],["Target ACC","Sample ACC"] ,display_id=3)
	print(pred_sample)
	print(sample_labels)
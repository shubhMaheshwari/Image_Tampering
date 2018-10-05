# Main module to train the model, load the data,
# do gradient descent etc.
from dataloader import DataSet,collate_fn
from model import Model
from options import TrainOptions
import torch
from torchvision.transforms import *
import torch.optim as optim
from visualize import Visualizer
import numpy as np 

# Get the Hyperparaeters 
opt = TrainOptions().parse()
train_data = DataSet(opt)
train_loader = torch.utils.data.DataLoader(train_data,collate_fn=collate_fn,
				batch_size=opt.batch_size,shuffle=True,num_workers=2)
val_loader = torch.utils.data.DataLoader(train_data,collate_fn=collate_fn,
				batch_size=10*opt.batch_size,shuffle=True,num_workers=2)

# Load the model
model = Model(opt)
if opt.use_gpu:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.cuda()	
	model = torch.nn.DataParallel(model, device_ids=opt.gpus)
	
print('------------ Model -------------')
print(model)
print('-------------- End ----------------')	


# Loss functons
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=opt.lr,weight_decay=opt.weight_decay,momentum=opt.momentum)

# Visualizer(Visdom)
vis = Visualizer(opt)
loss_list = []
# Training loop
for epoch in range(opt.epoch):
	model.train()
	for i,data in enumerate(train_loader,0):

		true_im,true_labels,fake_im,fake_labels = data
		pred_true,pred_false,loss_mmd = model(true_im.cuda(),fake_im.cuda())

		loss_true = criterion(pred_true,true_labels.cuda())
		loss_fake = criterion(pred_false,fake_labels.cuda())

		loss = opt.lambda_true*loss_true + opt.lambda_fake*loss_fake + opt.lambda_mmd*loss_mmd

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()	
		loss_list.append(loss.data)
		if i % opt.print_iter == 0:

			pred_true = np.argmax(pred_true.cpu().data.numpy(),axis=1)
			acc = np.mean(pred_true == true_labels.cpu().data.numpy())

			pred_false = np.argmax(pred_false.cpu().data.numpy(),axis=1)
			acc += np.mean(pred_false == fake_labels.data.numpy())


			acc = acc/2
			for j,val_data in enumerate(val_loader,0):
				true_im,true_labels,fake_im,fake_labels = val_data
				val_true,val_false,_ = model(true_im.cuda(),fake_im.cuda())
				val_true = np.argmax(val_true.cpu().data.numpy(),axis=1)
				val_false = np.argmax(val_false.cpu().data.numpy(),axis=1)
				val_acc = np.mean(val_true == true_labels.cpu().data.numpy())
				val_acc += np.mean(val_false == fake_labels.cpu().data.numpy())
				val_acc = val_acc/2
				break

			print("Iter:{}/{} Loss:{} Acc:{} Val_acc:{}".format(i,len(train_loader), loss.data[0],acc,val_acc))
			# Plot the loss 
			vis.plot_errors(loss_list)

			# Accuracy	
		if i > opt.iter:
			break
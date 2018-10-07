# Test  the model, load the data,
# followed by saving our model for later testing
from dataloader import DataSet,collate_fn
from model import Model
from options import TestOptions
import torch
from torchvision.transforms import *
import numpy as np 
import os
# Get the Hyperparaeters 
opt = TestOptions().parse()

# Load the sample dataset(COCO)
sample_dataset = DataSet(opt,opt.sample_true_dir,opt.sample_fake_dir)
sample_loader = torch.utils.data.DataLoader(sample_dataset,collate_fn=collate_fn,
				batch_size=opt.test_cases,num_workers=2,shuffle=True)

# Same method for target dataset(CASIA V2)
target_dataset = DataSet(opt,opt.target_true_dir,opt.target_fake_dir)
target_loader = torch.utils.data.DataLoader(target_dataset,collate_fn=collate_fn,
				batch_size=opt.test_cases,num_workers=2,shuffle=True)


# Load the model and send it to gpu
model = Model(opt)
if opt.use_gpu:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.cuda()	
	model = torch.nn.DataParallel(model, device_ids=opt.gpus)

# Load the weights and make predictions
model.load_state_dict(torch.load('./checkpoints/' + 'model_{}.pt'.format(opt.epoch)))

# Print our model 
print('------------ Model -------------')
print(model)
print('-------------- End ----------------')

model.eval()
sample_images, sample_labels = next(iter(sample_loader))
target_images, target_labels = next(iter(target_loader))
pred_sample,pred_target,loss_mmd = model(sample_images.cuda(),target_images.cuda())	

pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )

sample_acc = np.mean(pred_sample == sample_labels.cpu().data.numpy())

pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)
target_acc = np.mean(pred_target == target_labels.data.numpy())

print("==============Best results======================")
print(pred_sample)
print(sample_labels)
print("Validation Sample_Acc:{} Target_Acc:{}".format(sample_acc,target_acc))

# Test  the model, load the data,
# followed by saving our model for later testing
from dataloader import DataSet,collate_fn
from model import Model
from options import TestOptions
import torch
from torchvision.transforms import *
import numpy as np 
import os
from sklearn.metrics import confusion_matrix
# Get the Hyperparaeters 
opt = TestOptions().parse()

# Load the sample dataset(COCO)
sample_dataset = DataSet(opt,opt.sample_true_dir,opt.sample_fake_dir,opt.sample_fake_dir_mask,opt.sample_dataset)
sample_loader = torch.utils.data.DataLoader(sample_dataset,collate_fn=lambda x: collate_fn(x, -1),
				batch_size=1,num_workers=2,shuffle=True)

# Same method for target dataset(CASIA V2)
target_dataset = DataSet(opt,opt.target_true_dir,opt.target_fake_dir,opt.target_fake_dir_mask,opt.target_dataset)
target_loader = torch.utils.data.DataLoader(target_dataset,collate_fn=lambda x: collate_fn(x, -1),
				batch_size=1,num_workers=2,shuffle=True)


# Load the model and send it to gpu

device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")
opt.device = device
model = Model(opt)
if opt.use_gpu:

	model = model.cuda()	
	model = torch.nn.DataParallel(model, device_ids=opt.gpus)

# Load the weights and make predictions
model.load_state_dict(torch.load('./checkpoints/' + 'model_{}.pt'.format(opt.load_epoch)))

# Print our model 
print('------------ Model -------------')
print(model)
print('-------------- End ----------------')

model.eval()

K = opt.best_k


sample_test_list = []
pred_sample_test_list = []
target_test_list = []
pred_target_test_list = []
for i in range(opt.test_cases):
	sample_images, sample_labels = next(iter(sample_loader))
	target_images, target_labels = next(iter(target_loader))

	pred_sample,pred_target,loss_mmd = model(sample_images.to(device),target_images.to(device))	
	pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )
	pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)


	sample_labels = np.sum(sample_labels.numpy(),0) 
	pred_sample = np.sum(pred_sample,0)
	print("Misclassied {} from Sample Image ".format(abs(pred_sample-sample_labels)))

	target_labels = np.sum(target_labels.numpy(),0)
	pred_target = np.sum(pred_target,0)
	print("Misclassied {} from Target Image ".format(abs(pred_target-target_labels)))

	sample_test_list.append(np.sign(sample_labels))
	pred_sample_test_list.append( ((pred_sample - K) > 0).astype('int'))
	target_test_list.append(np.sign(target_labels))
	pred_target_test_list.append(( (pred_target -K) > 0).astype('int'))

sample_test_list = np.array(sample_test_list)
pred_sample_test_list = np.array(pred_sample_test_list).T
sample_acc = np.mean(sample_test_list == pred_sample_test_list)

target_test_list = np.array(target_test_list)
pred_target_test_list = np.array(pred_target_test_list).T

target_acc = np.mean(target_test_list == pred_target_test_list)

conf_target = confusion_matrix(target_test_list,pred_target_test_list)

print("==============Best results======================")
print(conf_target)
print("Validation Sample_Acc:{} Target_Acc:{}".format(sample_acc,target_acc))

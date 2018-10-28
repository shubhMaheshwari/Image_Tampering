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

target_path_list = [("IEEE","../IEEE/training/phase-01/training/pristine/","../IEEE/training/phase-01/training/masks/","../IEEE/training/phase-01/training/fake/")]

target_loaders =[]
for paths in target_path_list:
	dataset_name,true_path,mask_path,fake_path = paths
	dataset = DataSet(opt,true_path,fake_path,mask_path,dataset_name)

	loader = torch.utils.data.DataLoader(dataset,collate_fn= lambda x: collate_fn(x, -1),batch_size=1,num_workers=1)

	target_loaders.append(loader)	



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


target_test_list = []
pred_target_test_list = []

for target_loader in target_loaders:
	for i in range(opt.test_cases):
		target_images, target_labels,percent_list = next(iter(target_loader))

		_,pred_target,loss_mmd = model(target_images.to(device),target_images.to(device))	
		pred_target = np.argmax(pred_target.cpu().data.numpy(),axis=1)



		target_labels = np.sum(target_labels.numpy(),0)
		pred_target = np.sum(pred_target,0)
		print("Misclassied {} from Target Image ".format(abs(pred_target-target_labels)))

		target_test_list.append(np.sign(target_labels))
		pred_target_test_list.append(( (pred_target -K) > 0).astype('int'))


	target_test_list = np.array(target_test_list)
	pred_target_test_list = np.array(pred_target_test_list).T

	target_acc = np.mean(target_test_list == pred_target_test_list)

	conf_target = confusion_matrix(target_test_list,pred_target_test_list)

	print("==============Best results======================")
	print(conf_target)
	print("Validation Target_Acc:{}".format(target_acc))

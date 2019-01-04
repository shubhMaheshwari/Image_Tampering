# Test  the model, load the data,
# followed by saving our model for later testing
from dataloader import DataSet
from model import Model
from options import TestOptions
import torch
from torchvision.transforms import *
import numpy as np 
import os
from sklearn.metrics import confusion_matrix,roc_curve
import cv2
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
# Get the Hyperparaeters 
opt = TestOptions().parse()

test_dir = "./CASIA_test_patches/"
target_dataset = DataSet(opt,test_dir)
target_loader = torch.utils.data.DataLoader(target_dataset,batch_size=opt.val_batch_size,num_workers=30,shuffle=False)

# Load the model and send it to gpu
test_transforms =  transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
						 std = [ 1/0.229, 1/0.224, 1/0.225 ]),
	transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
						 std = [ 1., 1., 1. ]) ])


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

def get_accuracy(pred_sample, sample_labels):
	confidence = pred_sample[np.arange(len(sample_labels)),sample_labels[0,:]]
	# confidence = pred_sample[:,1]
	pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )

	return pred_sample,confidence

classified_list = []
confidence_list = []
label_list = []
results = {}
for target_images, target_labels,fileno,patch_y,patch_x,total_y,total_x in target_loader:
	target_labels.squeeze(1)
	fileno  = fileno.numpy().astype('int')
	patch_y = patch_y.numpy().astype('int')
	patch_x = patch_x.numpy().astype('int')
	total_y = total_y.numpy().astype('int')
	total_x = total_x.numpy().astype('int')

	try:
		pred_target,_,loss_mmd = model(target_images.to(device),target_images.to(device))
	except RuntimeError  as r:
		print("Error:",r)
		torch.cuda.empty_cache()
		continue

	target_pred,confidence = get_accuracy(pred_target,target_labels)	
	classified_list.extend(target_pred)
	label_list.extend(target_labels)
	confidence_list.extend(confidence.cpu().data.numpy())
	print("Creating Images")
	for i in range(target_images.shape[0]):

		if fileno[i] not in results:
			results[fileno[i]] = {'pred': np.zeros((total_y[i],total_x[i])), 'gr': np.zeros((total_y[i],total_x[i])), 
				'completed':0}

		try:
			results[fileno[i]]['pred'][patch_y[i],patch_x[i]] = target_pred[i]	
			results[fileno[i]]['gr'][patch_y[i],patch_x[i]] = target_labels[i].cpu().data.numpy() 
			results[fileno[i]]['completed'] = results[fileno[i]]['completed'] +1 
		except Exception as e:
			print("BT MAX:",fileno[i],patch_y[i],patch_x[i], total_y[i],total_x[i],results[fileno[i]]['pred'].shape)
			print(e)

K_list_gr = []
K_list_pred = []

for fileno in results:


	w,h = results[fileno]['pred'].shape

	# gr_im = np.zeros((64*h,64*w,3),dtype='int')
	# pred_im = np.zeros((64*h,64*w,3),dtype='int')
	# xxx = False
	# for x in range(w):
	# 	for y in range(h):
	# 		filename = "{}_{}_{}_{}_{}_{}_{}.png".format('CASIA',fileno,x,y, h,w ,int(results[fileno]['gr'][x][y]))
	# 		print(filename)
	# 		im = cv2.imread(os.path.join(test_dir,filename))
		
	# 		pred_im[64*y: 64*y + 64, 64*x: 64*x+64,:] = im
	# 		gr_im[64*y: 64*y + 64, 64*x: 64*x+64,:] = im


	# 		if results[fileno]['gr'][x][y] == 1.0:
	# 			gr_im[64*y: 64*y + 64, 64*x: 64*x+64,0] += 127				

	# 		print("What??")
	# 		if results[fileno]['pred'][x][y] == 1 and xxx == False: 
	# 			pred_im[64*y: 64*y + 64, 64*x: 64*x+64,0] += 127	
	# 			xxx = True

	# fig = plt.figure()
	# ax1 = fig.add_subplot(1,2,1)
	# ax1.set_title('Prediction')
	# ax1.set_axis_off()
	# ax1.imshow(pred_im)

	# ax2 = fig.add_subplot(1,2,2)
	# ax2.set_title('Ground Truth')
	# ax1.set_axis_off()
	# ax2.imshow(gr_im)

	# fig.savefig(os.path.join('/media/shubh/My Passport/shubh/CASIA2/results',"{}.png".format(fileno)))
	# plt.close(fig)

	K_list_pred.append(np.sum(results[fileno]['pred']))
	K_list_gr.append(np.sum(results[fileno]['gr']))

fpr, tpr, thresholds = roc_curve(classified_list,confidence_list)
conf_target = confusion_matrix(label_list,classified_list)

print("==============Best results======================")
print(conf_target)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(tpr, fpr, marker='.')
plt.show()

K_list_gr = np.array(K_list_gr)
K_list_pred = np.array(K_list_pred) 

for k in range(0,100):
	conf_matrix = np.zeros((2,2))
	for i in range(len(K_list_gr)):
		if K_list_gr[i] > 0 and K_list_pred[i] > k:
			conf_matrix[1,1] += 1

		elif K_list_gr[i] > 0 and K_list_pred[i] <= k:
			conf_matrix[1,0] += 1

		elif K_list_gr[i] <= 0 and K_list_pred[i] > k:
			conf_matrix[0,1] += 1
		else:
			conf_matrix[0,0] += 1
	print(k,(conf_matrix[0,0]+conf_matrix[1,1])/200)
	print("Accuracy per image")
	print(conf_matrix)
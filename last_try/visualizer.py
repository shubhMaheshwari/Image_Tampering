import numpy as np
import os
import time
import visdom
import cv2
class Visualizer():
    def __init__(self, opt):

        self.opt = opt
        self.vis = visdom.Visdom()
        self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

        # Loss
        self.loss_list = []
        self.loss_sample_list = []
        self.loss_target_list = []
        self.loss_mmd_list = []

        # Accuracy
        self.target_acc_list = []
        self.train_target_acc_list = []
        self.sample_acc_list = []
        self.train_sample_acc_list = []

    def append_metrics(self,loss, loss_sample, loss_target, loss_mmd,target_acc, sample_acc):

        # Upate the loss list and plot it
        self.loss_list.append(loss.cpu().data.numpy())
        self.loss_sample_list.append(loss_sample.cpu().data.numpy())
        self.loss_target_list.append(loss_target.cpu().data.numpy())
        self.loss_mmd_list.append(loss_mmd.cpu().data.numpy())

        self.train_target_acc_list.append(target_acc)
        self.train_sample_acc_list.append(sample_acc)


    # errors: dictionary of error labels and values        
    def plot_graph(self,X,Y,labels,display_id,title,axis=['x','y']):
        Y = np.array(Y).T
        if X == None: 
            X = np.arange(0,Y.shape[0])
        
        self.vis.line(
            X=X,
            Y=Y,
            win = display_id,
            opts={
            'title': title,
            'legend': labels,
            'xlabel': axis[0],
            'ylabel': axis[1]}
            )

        return

    def plot_loss(self):
        self.plot_graph(None,[self.loss_list,self.loss_sample_list,self.loss_target_list,self.loss_mmd_list],["Loss","Sample Loss", "Target Loss", "Mmd Loss"] ,display_id=1,title=' loss over time',axis=['Epoch','Loss'])

    def plot_acc(self):
        self.plot_graph(None,[self.train_target_acc_list,self.train_sample_acc_list],["Train Target ACC","Train Sample ACC"] ,display_id=4,title='Training Accuracy',axis=['Epoch','Acc'])
    def show_image(self,img,y_pred,y,display_id,title):

        
        self.vis.image(
            img,
            win=display_id,
            opts={
            'caption': "Y:{} Y_pred:{}".format(y,y_pred),
            'title': title
            })
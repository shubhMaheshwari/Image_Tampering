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
    # errors: dictionary of error labels and values
    def plot_graph(self,X,Y,labels,display_id):
        
        Y = np.array(Y).T
        if X == None: 
            X = np.arange(0,Y.shape[0])
        
        self.vis.line(
            X=X,
            Y=Y,
            win = display_id,
            opts={
            'title': ' loss over time',
            'legend': labels,
            'xlabel': 'epoch',
            'ylabel': 'loss'}
            )

        return

    def show_image(self,img,y,y_pred,display_id):

        
        self.vis.image(
            img,
            win=display_id,
            opts={
            'caption': "Y:{} Y_pred:{}".format(y,y_pred)
            })
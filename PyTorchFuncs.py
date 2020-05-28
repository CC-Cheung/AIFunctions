import argparse
from time import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

# from model import MultiLayerPerceptron
# from dataset import AdultDataset
class NNHandler:
    model:nn.Module
    lossFunc:nn._Loss
    optimizer:optim.optimzer
    loaders = []
    lr:float
    epochs:int
    verbose:bool=False
    histories=[[],[]]
    #There's also self.correctness

    def loadModel(self, model, lossFunc, optimizer, lr):
        ######
        # 4.4 YOUR CODE HERE
        self.model = model
        self.lossFunc =lossFunc
        self.optimizer = optimizer
        self.lr=lr
        ######

    def loadData(self, batch_size,sets_of_data):

        """
        :param batch_size: iterable of sizes
        :param sets_of_data: iterable of datasets
        :return:
        """
        for i in range(len(sets_of_data)):
            self.loaders.append(data.DataLoader(sets_of_data[i], batch_size=batch_size[i], shuffle=True, num_workers=2))
    def loadCorrectness(self, correctness=lambda x,y: x==y):
        self.correctness=correctness


    def printAlt(self, arg):
        if self.verbose==True:
            print(arg)
    def evalLossesAccs(self, ind):
        X, y = next(iter(self.loaders[ind]))
        X = X.float()
        y = y.float()
        total_corr = 0

        # accuracy
        predict = self.model(X)
        corr = []
        for i in range(len(y)):
            corr.append(self.correctness(predict[i], y[i]))
            # print ("comparing", predict[i], y[i], "\n")
        acc = float(sum(corr)) / len(y)

        # loss
        loss = self.lossFunc(input=predict.squeeze(), target=y)
        print(loss.item(), acc)
        ######
        return [loss.item(), acc]
    def train(self, epochs, trainInd=0, evalInd=None, graphRate=10, printVerbose=False, graphVerbose=False):
        #TODO:evalInd to have test data too (maybe)
        """

        :param epochs:
        :param trainInd: default is first dataloader
        :param evalInd: default is second dataloader
        :param printVerbose:
        :return:
        """
        self.epochs=epochs
        self.verbose=printVerbose
        for i in range(epochs):
            self.printAlt("\n", i, "newepoch\n")
            for j, batch_data in enumerate(self.loaders[trainInd], 0):
                self.printAlt("batch#", j)
                X = batch_data[0]
                y = batch_data[1]
                self.optimizer.zero_grad()
                # https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
                X = X.float()
                y = y.float()
                predict = self.model(X)  # squeeze and relu reduce # of channel
                # print (predict.data)
                loss = self.lossFunc(input=predict.squeeze(), target=y)

                loss.backward()  # compute the gradients of the weights

                self.optimizer.step()  # this changes the weights and the bias using the learningrate and gradients

                #   compute the accuracy of the model on the validation data  (don't normally do this every epoch, but is OK here)

            # record data for plotting

            self.histories[0].append(self.evalLossesAccs(trainInd))
            if not evalInd is None:
                self.histories[1].append(self.evalLossesAccs(evalInd))

            if graphVerbose and i % graphRate == 0:
                t_losses_accs_arr = np.array(self.histories[0])
                self.printAlt(t_losses_accs_arr.shape)
                plt.figure()
                plt.plot(np.arange(0, i + 1), t_losses_accs_arr[:, 0])
                if not evalInd is None:
                    v_losses_accs_arr = np.array(self.histories[1])
                    plt.plot(np.arange(0, i + 1), v_losses_accs_arr[:, 0])

                plt.title('Training Loss vs. Epoch')
                plt.xlabel('epoch')
                plt.ylabel('Loss')
                plt.show()

                plt.figure()
                plt.plot(np.arange(0, i + 1), t_losses_accs_arr[:, 1])
                if not evalInd is None:
                    plt.plot(np.arange(0, i + 1), v_losses_accs_arr[:, 1])

                plt.title('Accuracy vs. Epoch')
                plt.xlabel('epoch')
                plt.ylabel('Accuracy')
                plt.show()
                #TODO: Add maybe time
        return 1
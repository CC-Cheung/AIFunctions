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

# from model import MultiLayerPerceptron
# from dataset import AdultDataset
class NNHandler:
    network:nn.Module
    def load_data(batch_size,sets_of_data):
        ######
        """
        :param batch_size: iterable of sizes
        :param sets_of_data: iterable of datasets
        :return:
        """
        loaders = []
        # 4.1 YOUR CODE HERE
        for i in range(len(sets_of_data)):
            loaders.append(data.DataLoader(sets_of_data[i], batch_size=batch_size[i], shuffle=True, num_workers=2))
        ######
        return loaders

    def train():
        for i in range(epochs):

            print("\n", i, "newepoch\n")
            for j, batch_data in enumerate(t_im_loader, 0):
                print("batch#", j)
                X = batch_data[0]
                y = batch_data[1]
                optimizer.zero_grad()
                # https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
                X = X.float()
                y = y.float()
                predict = model(X)  # squeeze and relu reduce # of channel
                # print (predict.data)
                loss = loss_fnc(input=predict.squeeze(), target=y)

                loss.backward()  # compute the gradients of the weights

                optimizer.step()  # this changes the weights and the bias using the learningrate and gradients

                #   compute the accuracy of the model on the validation data  (don't normally do this every epoch, but is OK here)

            # record data for plotting
            X, y = next(iter(t_im_loader_all))
            X = X.float()
            y = y.float()
            t_losses_accs.append(eval_losses_accs(model, X, y, loss_fnc))
            if (i % 10 == 0):
                t_losses_accs_arr = np.array(t_losses_accs)
                print(t_losses_accs_arr.shape)
                plt.figure()
                plt.plot(np.arange(0, i + 1), t_losses_accs_arr[:, 0])
                plt.title('Training Loss vs. Epoch')
                plt.xlabel('epoch')
                plt.ylabel('Loss')
                plt.show()

                plt.figure()
                plt.plot(np.arange(0, i + 1), t_losses_accs_arr[:, 1])
                plt.title('Accuracy vs. Epoch')
                plt.xlabel('epoch')
                plt.ylabel('Accuracy')
                plt.show()
                end = time()
                print(end, start, end - start)
        return 1
import matplotlib.pyplot as plt
import numpy as np

import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch
class simpleDataset(data.Dataset):
    def __init__(self, X, y):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
possibleRes=torch.as_tensor(np.array([0,1])).float()

class NNHandler:
    #TODO: add return information (history, epochs...)
    model:nn.Module
    lossFunc:nn.Module
    optimizer:optim
    lr:float
    epochs:int
    #There's also self.correctness
    def __init__(self):
        self.verbose:bool=False
        self.histories=[[],[]]
        self.loaders=[]
    def printAlt(self, *args):
        if self.verbose==True:
            print(*args)
    def loadModel(self, model, optimizer, lr, lossFunc=nn.MSELoss()):
        ######
        # 4.4 YOUR CODE HERE
        self.model = model
        self.optimizer = optimizer
        self.lossFunc =lossFunc
        self.lr=lr
        ######

    def loadData(self, sets_of_data,batch_size):
        #TODO: GPU mem pin
        """
        :param batch_size: iterable of sizes
        :param sets_of_data: iterable of datasets
        :return:
        """
        for i in range(len(sets_of_data)):
            self.loaders.append(data.DataLoader(sets_of_data[i], batch_size=batch_size[i], shuffle=True, num_workers=2))
    def loadCorrectness(self, correctness=lambda x,y: x==y):
        self.correctness=correctness

    def evalLossesAccs(self, ind):
        X, y = next(iter(self.loaders[ind]))
        X = X.float()
        y = y.float()
        total_corr = 0

        # accuracy
        predict = self.model(X)
        acc = float(sum(self.correctness(predict, y).float())) / len(y)

        # loss
        loss = self.lossFunc(input=predict.squeeze(), target=y)
        ######
        return [loss.item(), acc]
    def train(self, epochs, trainInd=0, evalInd=None,  printVerbose=False, graphVerbose=False,graphRate=10):
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
            self.printAlt(self.histories[0][-1])
            if not evalInd is None:
                self.histories[1].append(self.evalLossesAccs(evalInd))
                self.printAlt(self.histories[1][-1])

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
if __name__=="__main__":
    # myNNHandler=NNHandler()
    # possibleRes=torch.as_tensor(np.array([0,1])).float()
    #
    correctness=lambda x, y: torch.eq(torch.argmin((x - possibleRes).abs(), dim=1).float(), y)
    # myNNHandler.loadCorrectness(correctness())
    tttIn=np.random.randint(0,2,[100,9])
    tttOut=np.array([(tttIn[i,0] and tttIn[i,4] and tttIn[i,8]) or (tttIn[i,2] and tttIn[i,4] and tttIn[i,6]) for i in range(100)])
    # tttData=simpleDataset(tttIn,tttOut)

    model=nn.Linear(9,1)
    lr=0.1


    optimizer=optim.SGD(model.parameters(), lr)
    lossFunc=nn.MSELoss()
    # myNNHandler.loadModel(model,optimizer,lr)
    # myNNHandler.loadData([tttData],[len(tttData)])
    # print(myNNHandler.evalLossesAccs(0))
    #
    # myNNHandler.train(60,printVerbose=True, graphVerbose=False)
    X = torch.as_tensor(tttIn).float()
    y = torch.as_tensor(tttOut).float()
    for i in range(100):
        print("\n", i, "newepoch\n")

        optimizer.zero_grad()
        # https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

        predict = model(X)  # squeeze and relu reduce # of channel
        # print (predict.data)
        loss = lossFunc(input=predict.squeeze(), target=y)

        loss.backward()  # compute the gradients of the weights

        optimizer.step()  # this changes the weights and the bias using the learningrate and gradients

            #   compute the accuracy of the model on the validation data  (don't normally do this every epoch, but is OK here)

        # record data for plotting

        total_corr = 0

        # accuracy
        predict = model(X)
        acc = float(sum(correctness(predict, y).float())) / len(y)

        # loss
        loss = lossFunc(input=predict.squeeze(), target=y)
        ######

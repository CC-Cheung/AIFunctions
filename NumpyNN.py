import random
import math
import numpy as np
import matplotlib.pyplot as plt
"""
Chi-Chung Cheung 1004164968
Agent Description:
Approach, Deep Q-learning.
Q-value is kind of like the utility of being in a state. Deep Q-learning tries to approximate this value through a 
neural network. 
I implement a layer class, which includes the LinTrans as well as activation functions (I used logistic and relu), and
loss function(new class named so, used MSE). Layers store input, output, backgrad. LinTrans also stores gradients. 
Layers also have forward pass (calc results) and backward pass(calc gradients).    
A neural network is made of layers. I initialize through my personal description [4,8,'l',8,'r',2,'MSE]. l (for logistic)
and gradient clip norm because gradients kept exploding. I also highLossMode so the network is reset. Its 4 observation
dimensions, and the output q-values of left and right(2).
I store states in a circular buffer (10,000). I do mini-batch gradient descent. I train one mini-batch once about every 
4 time steps. The target is the q-value prediction of a state but the q-value of the action taken is the reward + gamma*
max q-value of nextstate. If it is terminal, then its just the reward. Punishment is increased to -5
If the numsteps reaches 420, I just cut off training just in case the gradients explode later. 

Referenced:
https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/
https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
https://github.com/VBot2410/Deep-Q-Learning-Cartpole
TODO: Insert a 10-15 line description (80 characters wide) of the algorithm
that you implemented in your agent. If you used a reference paper or book,
you may add additional lines to cite this reference.
"""

class Layer:
    def __init__(self):
        self.input=None
        self.output=None
        self.backGrad=None
        self.name=None
    def printLayer(self):
        print("\nlayer ", self.name)
        print("input=", self.input)
        print("output=", self.output)
        print("backGrad=", self.backGrad)

    def forward(self, input):
        raise NotImplementedError
    def backward(self, outGrad):
        raise NotImplementedError
class LinTrans(Layer):
    def __init__(self, inSize, outSize, lr=0.1):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = (xW) + b
        super().__init__()
        self.name="LinTrans in={}, out={}".format(inSize,outSize)
        self.gradW = None
        self.gradb = None
        self.lr = lr
        self.W = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (inSize + outSize)),
                                        size=(inSize, outSize))
        # self.W = np.random.uniform(-0.5,0.5,(inSize,outSize))
        # self.W=np.zeros([inSize,outSize])
        self.b = np.zeros(outSize)
        # self.W=np.nan*np.ones([inSize,outSize])
        # self.b = np.zeros(outSize)*np.nan
    def printLayer(self):
        super().printLayer()
        print("Weights+bias=", self.W, "\n",self.b)
        print("grads=", self.gradW,self.gradb)
    def forward(self, input):
        self.input=input
        self.output=input@self.W + self.b

        return self.output

    def updateGrad(self, outGrad):
        self.backGrad = outGrad @ self.W.T
        self.gradW = self.input.T @ outGrad/outGrad.shape[0]
        self.gradW=self.gradW/np.linalg.norm(self.gradW)
        self.gradb = outGrad.sum(axis=0)/outGrad.shape[0]
        self.gradb=self.gradb/np.linalg.norm(self.gradb)

        assert self.gradW .shape == self.W .shape and self.gradb.shape == self.b.shape
        return self.backGrad

    def updateWb(self):
        self.W = self.W - self.lr * self.gradW
        self.b = self.b  - self.lr * self.gradb
    def backward(self, outGrad):
        backGrad=self.updateGrad(outGrad)
        self.updateWb()
        return backGrad
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.name='ReLU'
    def forward(self, input):
        self.input=input
        self.output=(input)*(input>0)
        return self.output

    def backward(self, outGrad):
        # Compute gradient of loss w.r.t. ReLU input
        self.backGrad = outGrad*(self.input > 0)
        return self.backGrad
class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'Softmax'
    def forward(self, input):
        self.input = input
        self.output = np.exp(input)/np.sum(np.exp(input))##possible error
        return self.output
    def backward(self, outGrad):
        self.backGrad = outGrad * self.output.T * (1 - self.output)
        return self.backGrad
class Logistic(Layer):
    def __init__(self):
        super().__init__()
        self.name='Logistic'
    def forward(self, input):
        self.input=input
        self.output=1/(1+np.exp(-input))
        return self.output
    def backward(self, outGrad):
        self.backGrad=outGrad * self.output * (1-self.output)
        return self.backGrad
class LossFunc(Layer):
    def forward(self, pred,target):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError
class MSE(LossFunc):
    def __init__(self):
        super().__init__()
        self.name='MSE'
        self.target=None
    def forward(self, pred,target):
        self.input=pred
        self.output=np.mean(np.square(target-pred), axis=1)
        self.target=target
        return self.output
    def backward(self):
        self.backGrad=2*(self.input-self.target)/self.target.shape[1]
        return self.backGrad

ACT_LAYERS = {
  "r": ReLU,
  "l": Logistic,
  "s": Softmax
}
LOSS_LAYERS = {
  "MSE": MSE
}
nanRemove = np.vectorize(lambda x: x if x == x else np.random.rand() - 0.5)
zeroRevive = np.vectorize(lambda x: x if x != 0 else np.random.rand()/10 - 0.05)
def fixNan(a):
    temp=nanRemove(a)
    temp=zeroRevive(temp)

    return temp/np.linalg.norm(temp)
class NeuralNetwork:

    def __init__(self):
        self.layers=[]
        self.depth=0
    def __init__(self, descriptor,lr=0.01):
        self.layers = []
        inSize=descriptor[0]
        for i in descriptor[1:-1]:
            if type(i) is int:
                self.layers.append(LinTrans(inSize,i,lr=lr))
                inSize=i
            else:
                self.layers.append(ACT_LAYERS[i]())
        self.layers.append(LOSS_LAYERS[descriptor[-1]]())
        self.depth = len(self.layers)
        self.highLossMode=False
    def addLayer(self,Layer): #Manual layer creation
        self.layers.append(Layer)
        self.depth+=1
    def printLayers(self):
        print("NEURAL NETWORK")
        for i in self.layers:
            i.printLayer()
        print("-----------------------------------")
    def predict(self, input):

        pred = input
        for i in self.layers[:-1]:
            # i.printLayer()
            pred = i.forward(pred)
            # i.printLayer()

        return pred
    def forward(self, input, target): #returns loss
        return self.layers[-1].forward(self.predict(input),target)
    def computeAvgLoss(self, input, target):
        return np.average(self.forward(input,target))
    def backward(self):
        backGrad=self.layers[-1].backward()
        for i in self.layers[:-1][::-1]: #except for loss, do network backwards
            if self.highLossMode:
                backGrad=fixNan(backGrad)
            backGrad=i.backward(backGrad)

    def train(self, input, target):

        loss=self.forward(input,target)
        self.backward()

        return np.mean(loss)
    def setlr(self,lr):
        for i in self.layers:
            if "LinTrans" in i.name:
                i.lr=lr

    def printIfNan(self):
        for j in self.layers[0].W:
            for k in j:
                if k!=k:
                    self.printLayers()
                    return -1

    def highLossWbAdjust(self, lr):
        for i in self.layers:
            i.backGrad=fixNan(i.backGrad)
            i.input=fixNan(i.input)
            i.output=fixNan(i.output)
            if "LinTrans" in i.name:
                i.W=fixNan(i.W)
                i.b=fixNan(i.b)
                i.gradW = fixNan(i.gradW)
                i.gradb =fixNan(i.b)
                i.lr=lr
            elif "MSE" in i.name:
                i.target=fixNan(i.target)



if __name__=="__main__":
    np.random.seed(2)
    model = NeuralNetwork([9, 1, "MSE"], 0.05)
    # inputB = np.array([[-0.01090027, 0.04892414, 0.0235967, 0.04030543],[-0.01090027, 0.04892414, 0.0235967, 0.04030543]])
    # targetB = np.array([[0],[10]])
    # model.train(inputB,targetB)

    tttIn = np.random.randint(0, 2, [100, 9])
    tttOut = np.expand_dims(np.array(
        [(tttIn[i, 0] and tttIn[i, 4] and tttIn[i, 8]) or (tttIn[i, 2] and tttIn[i, 4] and tttIn[i, 6]) for i in
         range(100)]), axis=1)
    for i in range (1000000):
        print(model.train(tttIn,tttOut))
        print(np.sum((model.predict(tttIn)<0.5)==(tttOut<0.5))/100)
    # model.highLossWbAdjust(1)
    # model.train(inputB,targetB)
    # model.printLayers()


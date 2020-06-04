from nn_handler import NNHandler, FastDataLoader
from collections import deque
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch

def accumMult(a):
    total=1
    for i in a:
        total*=i
    return total
class DynamicTDS(data.TensorDataset): #stands for Dynamic TensorDataset
    def __init__(self, max_mem):

        self.empty=True
        self.max_mem=max_mem
    def add_data(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        if self.empty:
            self.empty=False
            self.tensors = tensors
        assert len(tensors)==len(self.tensors)
        assert all(self.tensors[i].size(1)==tensors[i].size(1) for i in range (len (tensors)))
        for i in len(tensors):
            self.tensors[i]=torch.cat((self.tensors[i],i), 0)

#TODO: Add memory (tensor state, reward, next, number for action), Add adding obs, override load data with take from memory, override train
class DQNHandler(NNHandler):
    def __init__(self, observation_space, action_space, MLPDesc, batchSize=64, gamma=0.995,lr=0.007, max_mem=10000):
        super().__init__()
        self.custom_load_model(MLPDesc)
        self.load_correctness(lambda x,y: abs(x-y))
        self.batchSize=batchSize
        self.gamma=gamma

        #uneeded
        # self.inSize = accumMult(observation_space.shape)
        # assert self.inSize==MLPDesc[0], "insize {} !=observation total dimension {}".format(self.inSize,MLPDesc[0])
        # self.observation_space=observation_space
        #
        # self.outSize = action_space.n
        # assert self.outSize == MLPDesc[-2], "outsize {} !=action total dimension {}".format(self.outSize, MLPDesc[-2])
        # self.action_space=action_space

        self.load_data([DynamicTDS(max_mem)],[batchSize])
        self.stop=False

    def train(self, num_step):

        states, actions, rewards, next_states, terminals=next(iter(self.loaders[0]))#TODO: add more efficient method

        with torch.no_grad():
            labels_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (self.gamma * labels_next * (1 - terminals))

        if self.stop:
            return -1
        self.curLoss= self.model.train(states, labels)

    def predict(self, state):
        return self.model(state)

    def update(self, *args):
        #state, action, reward, nextState, terminal, numStep

        new_args=[torch.as_tensor(args).unsqueeze(0) for i in range (len(args))]

        if new_args[4]:
            if new_args[5] > 500:
                new_args[3] *= -5
            else:
                new_args[3] *= 5

        self.loaders[0].dataset.add_data(*(new_args[:5]))


        if self.memEnd < self.batchSize or torch.rand(1) > 0.25:
            return -1

        self.curLoss = self.train(new_args[5])
        return self.curLoss
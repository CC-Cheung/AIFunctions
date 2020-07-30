import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


# from https://github.com/pytorch/pytorch/issues/15849
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

    def add_data(self, new_data):
        print(self.dataset)
        self.dataset = data.dataset.ConcatDataset([self.dataset, new_data])
        print(self.dataset)
#TODO: fix the messed up notation
ARCHITECTURES = {
    "lin":nn.Linear,
    "conv": nn.Conv2d,
    "lstm": nn.LSTM
}
ACTIVATIONS = {
    "r": nn.ReLU,
    "sig": nn.Sigmoid,
    "sof": nn.Softmax
}
LOSSES = {
    "MSE": nn.MSELoss,
    "CE": nn.CrossEntropyLoss,
    "BCELog": nn.BCEWithLogitsLoss
}
OPTIMIZERS = {
    "SGD": optim.SGD,
    "ADAM": optim.Adam
}


class NNHandler:
    # TODO: add return information (history, epochs...)
    model: nn.Module
    loss_func: nn.Module
    optimizer: optim
    lr: float
    epochs: int

    # There's also self.correctness
    def __init__(self):
        self.verbose: bool = False
        self.histories = [[], []]
        self.loaders = []
        self.clip_grad = False

    @classmethod
    def complete_init(cls, sets_of_data, batch_size, model, loss_func, optimizer, lr, correctness):
        myNNHandler = cls()
        myNNHandler.load_data(sets_of_data, batch_size)
        myNNHandler.load_model(model, loss_func, optimizer, lr)
        myNNHandler.load_correctness(correctness)

        return myNNHandler
    @classmethod
    def custom_nn(cls, descriptor):
        model = nn.Sequential()
        in_size = descriptor[0]
        depth = 1
        for i in descriptor[1:]:
            if type(i) is int:
                model.add_module("{}: linear {},{}".format(depth, in_size, i), nn.Linear(in_size, i))
                in_size = i
            else:
                model.add_module("{}: {}".format(depth, i), ACTIVATIONS[i]())
            depth += 1
        return model
    def print_alt(self, *args):
        if self.verbose == True:
            print(*args)

    def load_data(self, sets_of_data, batch_size):
        # TODO: GPU mem pin
        """
        :param batch_size: iterable of sizes
        :param sets_of_data: iterable of data.Datasets
        :return:
        """
        for i in range(len(sets_of_data)):
            self.loaders.append(FastDataLoader(sets_of_data[i], batch_size=batch_size[i], shuffle=True, num_workers=2))

    def custom_load_model(self, descriptor):
        """
        :param descriptor: [model stuff, loss_func, optimizer, lr]
        :return:
        """
        self.model = NNHandler.custom_nn(descriptor[:-3])
        self.loss_func = LOSSES[descriptor[-3]]()
        self.lr = descriptor[-1]
        self.optimizer = OPTIMIZERS[descriptor[-2]](self.model.parameters(), self.lr)

    def load_model(self, model, loss_func, optimizer, lr):
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.loss_func = loss_func


    def load_correctness(self,correctness):
        self.correctness = correctness

    def eval_losses_accs(self, ind):  # requires correctness
        X, y = next(iter(self.loaders[ind]))
        predict = self.model(X)

        # accuracy
        if self.correctness is None:
            acc=-1
        else:
            acc = float(sum(self.correctness(predict, y).float())) / len(y)
        # loss
        loss = self.loss_func(input=predict, target=y)

        return [loss.item(), acc]

    def train(self, epochs, trainInd=0, evalInd=None, print_verbose=False, graph_verbose=False, graphRate=10):
        # TODO:evalInd to have test data too (maybe)
        """

        :param epochs:
        :param trainInd: default is first dataloader
        :param evalInd: default is second dataloader
        :param print_verbose:
        :return:
        """
        self.epochs = epochs
        self.verbose = print_verbose
        for i in range(epochs):
            self.print_alt("\n", i, "newepoch\n")
            for j, batch_data in enumerate(self.loaders[trainInd], 0):
                self.print_alt("batch#", j)
                X = batch_data[0]
                y = batch_data[1]
                self.optimizer.zero_grad()
                # https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

                predict = self.model(X)  # squeeze and relu reduce # of channel
                loss = self.loss_func(input=predict, target=y)
                if self.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                loss.backward()  # compute the gradients of the weights

                self.optimizer.step()

            # record data for plotting

            self.histories[0].append(self.eval_losses_accs(trainInd))
            self.print_alt(self.histories[0][-1])
            if not evalInd is None:
                self.histories[1].append(self.eval_losses_accs(evalInd))
                self.print_alt(self.histories[1][-1])

            if graph_verbose and i % graphRate == 0:
                t_losses_accs_arr = np.array(self.histories[0])
                self.print_alt(t_losses_accs_arr.shape)
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
                # TODO: Add maybe time

    def save_to_file(self, path_name, continue_train=True):
        if continue_train:
            state = {
                'lr': self.lr,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),

            }
            torch.save(state, path_name)
        else:
            torch.save(self.model, path_name)

    def load_from_file(self, path_name, continue_train=True):
        if continue_train:
            state = torch.load(path_name)
            self.lr = state['lr']
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.histories = [[], []]
        else:
            torch.save(self.model, path_name)

    def print_model(self):
        batch_in = next(iter(self.loaders[0]))[0]
        summary(self.model, batch_in.shape)


if __name__ == "__main__":
    possibleRes = torch.as_tensor(np.array([0, 1])).float()

    tttIn = np.random.randint(0, 2, [100, 9])
    tttOut = np.array(
        [[(tttIn[i, 0] and tttIn[i, 4] and tttIn[i, 8]) or (tttIn[i, 2] and tttIn[i, 4] and tttIn[i, 6]) for i in
          range(100)]]).transpose()
    # tttOutNot = (tttOut + 1) % 2
    # tttOut = np.concatenate([tttOut, tttOutNot], axis=0).transpose()
    tttData = data.TensorDataset(torch.as_tensor(tttIn).float(), torch.as_tensor(tttOut).float())

    model = nn.Linear(9, 1)
    print(model(torch.from_numpy(np.array([i for i in range(9)])).float()))

    lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr)
    loss_func = nn.MSELoss()
    correctness = lambda x, y: torch.eq(torch.argmin((x - possibleRes).abs(), dim=1,keepdim=True).float(), y)
    myNNHandler = NNHandler.complete_init([tttData], [len(tttData)], model, loss_func, optimizer, lr,
                                          correctness)
    # myNNHandler.add_data(tttData, 0)
    myNNHandler.custom_load_model([9, 5, 'r',5, 1, 'MSE', 'ADAM', 0.005])
    print(myNNHandler.model)
    # c=iter(myNNHandler.model.parameters())
    # a=next(c)
    # a.data*=2
    # a=next(c)
    myNNHandler.train(60, print_verbose=True, graph_verbose=True)
    # myNNHandler.save_to_file('model.pt')
    #
    # myNNHandler.load_from_file('model.pt')
    # print(myNNHandler.eval_losses_accs(0))
    # myNNHandler.train(60, print_verbose=True, graph_verbose=True)
    myNNHandler.print_model()

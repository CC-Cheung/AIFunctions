from nn_handler import NNHandler, FastDataLoader
import torch.utils.data as data

import torch
import torch.nn.utils as utils
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
SUCCESS_REWARD = 195
SUCCESS_STREAK = 100
MAX_EPISODES = 200
MAX_STEPS = 5000
import sys, os


def accumMult(a):
    total = 1
    for i in a:
        total *= i
    return total


class DynamicTDS(data.TensorDataset):  # stands for Dynamic TensorDataset
    def __init__(self, max_mem):
        self.empty = True
        self.max_mem = max_mem
    def __len__(self):
        if self.empty:
            return 1 #Fake
        else:
            return self.tensors[0].size(0)
    def add_data(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        if self.empty:
            self.empty = False
            self.tensors = list(tensors)
            return
        assert len(tensors) == len(self.tensors)
        for i in range(len(tensors)):
            self.tensors[i] = torch.cat((self.tensors[i], tensors[i]), 0)


class DQNHandler(NNHandler):
    def __init__(self, observation_space, action_space, MLPDesc, batchSize=64, gamma=0.995, max_mem=10000,
                 expl_decay=0.993, expl_min=0.1, clip_grad=False): #TODO make use of obs and act spaces
        super().__init__()
        self.custom_load_model(MLPDesc)
        self.load_correctness(lambda x, y: abs(x - y))
        self.batchSize = batchSize
        self.gamma = gamma
        self.expl_decay = expl_decay
        self.expl_min = expl_min
        self.expl_rate=1
        self.clip_grad=clip_grad


        self.load_data([DynamicTDS(max_mem)], [batchSize])
        self.stop = False

    def load_data(self, sets_of_data, batch_size):
        # TODO: Change it to 1 loader
        """
        :param batch_size: iterable of sizes
        :param sets_of_data: iterable of data.Datasets
        :return:
        """
        for i in range(len(sets_of_data)):
            self.loaders.append(FastDataLoader(sets_of_data[i], batch_size=batch_size[i], shuffle=True)) #TODO: removed workers

    def train(self):

        states, actions, rewards, next_states, terminals = next(
            iter(self.loaders[0]))  # TODO: add more efficient method
        labels=self.model(states)
        max_next=self.model(next_states).detach().max(1)[0] #0=val, 1=index
        for i in range(labels.shape[0]):
            labels[i, actions[i]]= rewards[i]+self.gamma*max_next[i]*(1-terminals[i])

        self.optimizer.zero_grad()
        if self.clip_grad:
            utils.clip_grad_norm_(self.model.parameters(),self.clip_grad)
        # utils.clip_grad_value_(self.model.parameters(),1)
        predict = self.model(states)  # squeeze and relu reduce # of channel
        loss = self.loss_func(input=predict, target=labels)
        loss.backward()  # compute the gradients of the weights
        self.optimizer.step()  # this changes the weights and the bias using the learningrate and gradients
        return loss.data

    def predict(self, state):
        return self.model(state)
    def reset(self):
        pass
    def update(self, *args):
        # state, action, reward, nextState, terminal, numStep

        new_args = [torch.as_tensor(args[i]).unsqueeze(0).float() for i in range(len(args))]
        new_args[1]=new_args[1].int()

        if new_args[5]==0:
            self.expl_rate*=self.expl_decay

        self.loaders[0].dataset.add_data(*(new_args[:5]))

        if np.random.rand() > 0.25 or len(self.loaders[0].dataset)<self.batchSize:
            return -1

        self.curLoss = self.train()
        return self.curLoss

    def action(self, state):
        if np.random.rand()> max(self.expl_rate, self.expl_min):
            with torch.no_grad():
                return self.model(torch.as_tensor(state).unsqueeze(0).float()).detach().max(1)[1].numpy()[0]
        else:
            return np.random.randint(0,2)

class DDQNHandler(DQNHandler):
    def __init__(self, observation_space, action_space, MLPDesc, batchSize=64, gamma=0.995, max_mem=10000,
                 expl_decay=0.993, expl_min=0.1, clip_grad=False, tau=0.2,target_model_update_rate=10):
        super().__init__(observation_space, action_space, MLPDesc, batchSize=batchSize, gamma=gamma, max_mem=max_mem,
                 expl_decay=expl_decay, expl_min=expl_min, clip_grad=clip_grad)
        self.tau=tau
        self.target_model=self.custom_nn(MLPDesc[:-3])
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model_update_rate=target_model_update_rate
        self.num_updates=0
    def update(self, *args):
        self.num_updates+=1
        return super().update(*args)
    def train(self):
        states, actions, rewards, next_states, terminals = next(
            iter(self.loaders[0]))  # TODO: add more efficient method

        labels = self.model(states)
        max_next = self.model(next_states).detach().max(1)[1]  # 0=val, 1=index
        max_next= self.target_model(next_states)[torch.arange(states.shape[0]),max_next]
        for i in range(labels.shape[0]):
            labels[i, actions[i]] = rewards[i] + self.gamma * max_next[i] * (1 - terminals[i])

        self.optimizer.zero_grad()
        if self.clip_grad:
            utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        # utils.clip_grad_value_(self.model.parameters(),1)
        predict = self.model(states)  # squeeze and relu reduce # of channel
        # print (predict.data)
        loss = self.loss_func(input=predict, target=labels)
        loss.backward()  # compute the gradients of the weights
        self.optimizer.step()  # this changes the weights and the bias using the learningrate and gradients
        if self.num_updates % self.target_model_update_rate==0:
            #https://github.com/xkiwilabs/DQN-using-PyTorch-and-ML-Agents/blob/master/dqn_agent.py
            for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        return loss.data

def run_cart_pole():
    """
    Run instances of cart-pole gym and tally scores.

    The function runs up to 1,000 episodes and returns when the 'success'
    criterion for the OpenAI cart-pole task (v0) is met: an average reward
    of 195 or more over 100 consective episodes.
    """
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_STEPS

    # Create an instance of the agent.
    cp_agent = DDQNHandler(env.observation_space, env.action_space, [4, 8, 'sig', 8,'sig', 2, 'MSE', 'SGD', 0.08])
    avg_reward, win_streak = (0, 0)
    rewards = []
    losses = []
    exp = []
    exit = False
    count = 0
    for episode in range(1000):
        state = env.reset()

        # Reset the agent, if desired.
        cp_agent.reset()
        episode_reward = 0
        exp.append(max(cp_agent.expl_rate, cp_agent.expl_min))
        print(count)

        # The total number of steps is limited (to avoid long-running loops).
        for steps in range(MAX_STEPS):
            env.render()
            count += 1
            action = cp_agent.action(state)

            state_next, reward, terminal, info = env.step(action)
            if terminal:
                reward *= 5

            # Update any information inside the agent, if desired.

            losses.append(cp_agent.update(state, action, reward, state_next, terminal,steps))
            if losses[-1] > 1000000:
                print("high")

            episode_reward += reward  # Total reward for this episode.
            state = state_next

            if terminal:
                # Update average reward.
                if episode < SUCCESS_STREAK:
                    rewards.append(episode_reward)
                    avg_reward = float(sum(rewards)) / (episode + 1)
                else:
                    # Last set of epsiodes only (moving average)...
                    rewards.append(episode_reward)
                    # rewards.pop(0)
                    avg_reward = float(sum(rewards)) / SUCCESS_STREAK

                # Print some stats.
                print("Episode: " + str(episode) + \
                      ", Reward: " + str(episode_reward) + \
                      ", Avg. Reward: " + str(avg_reward) + \
                      ", exp rate " + str(exp[-1]))

                # Is the agent on a winning streak?
                if reward >= SUCCESS_REWARD:
                    win_streak += 1
                else:
                    win_streak = 0
                break
        print(losses[-1])

        if exit:
            break
        # print(rewards)

        # Has the agent succeeded?
        if win_streak == SUCCESS_STREAK and avg_reward >= SUCCESS_REWARD:
            return episode + 1, avg_reward

    fig, ax = plt.subplots()
    ax.plot(losses)

    fig, ax = plt.subplots()
    ax.plot(rewards)

    fig, ax = plt.subplots()
    ax.plot(exp)

    plt.show()
    # print(cp_agent.QSolver.memEnd)

    env.close()
    # Worst case, agent did not meet criterion, so bail out.
    return episode + 1, avg_reward


if __name__ == "__main__":
    # np.random.seed(2)
    episodes, best_avg_reward = run_cart_pole()
    print("--------------------------")
    print("Episodes to solve: " + str(episodes) + \
          ", Best Avg. Reward: " + str(best_avg_reward))
    print('enter forloop')

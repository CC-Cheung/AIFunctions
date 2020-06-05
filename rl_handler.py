from nn_handler import NNHandler, FastDataLoader
import torch.utils.data as data

import torch
import gym

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

    def add_data(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        if self.empty:
            self.empty = False
            self.tensors = tensors
        assert len(tensors) == len(self.tensors)
        assert all(self.tensors[i].size(1) == tensors[i].size(1) for i in range(len(tensors)))
        for i in len(tensors):
            self.tensors[i] = torch.cat((self.tensors[i], i), 0)


# TODO: Add memory (tensor state, reward, next, number for action), Add adding obs, override load data with take from memory, override train
class DQNHandler(NNHandler):
    def __init__(self, observation_space, action_space, MLPDesc, batchSize=64, gamma=0.995, max_mem=10000,
                 expl_decay=0.99, expl_min=0.1):
        super().__init__()
        self.custom_load_model(MLPDesc)
        self.load_correctness(lambda x, y: abs(x - y))
        self.batchSize = batchSize
        self.gamma = gamma
        self.expl_decay = expl_decay
        self.expl_min = expl_min
        self.expl_rate=1
        # uneeded
        # self.inSize = accumMult(observation_space.shape)
        # assert self.inSize==MLPDesc[0], "insize {} !=observation total dimension {}".format(self.inSize,MLPDesc[0])
        # self.observation_space=observation_space
        #
        # self.outSize = action_space.n
        # assert self.outSize == MLPDesc[-2], "outsize {} !=action total dimension {}".format(self.outSize, MLPDesc[-2])
        # self.action_space=action_space

        self.load_data([DynamicTDS(max_mem)], [batchSize])
        self.stop = False

    def train(self, num_step):

        states, actions, rewards, next_states, terminals = next(
            iter(self.loaders[0]))  # TODO: add more efficient method

        with torch.no_grad():
            labels_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (self.gamma * labels_next * (1 - terminals))

        if self.stop:
            return -1
        self.curLoss = self.model.train(states, labels)

    def predict(self, state):
        return self.model(state)

    def update(self, *args):
        # state, action, reward, nextState, terminal, numStep

        new_args = [torch.as_tensor(args).unsqueeze(0) for i in range(len(args))]

        if new_args[4]:
            if new_args[5] > 500:
                new_args[3] *= -5
            else:
                new_args[3] *= 5
        if new_args[5]==0:
            self.expl_rate*=self.expl_decay

        self.loaders[0].dataset.add_data(*(new_args[:5]))

        if self.memEnd < self.batchSize or torch.rand(1) > 0.25:
            return -1

        self.curLoss = self.train(new_args[5])
        return self.curLoss

    def action(self, state):
        if torch.rand(1)> max(self.expl_rate, self.expl_min)
            with torch.no_grad():
                return self.model(torch.as_tensor(state).unsqueeze(0)).detach().max(1)[0].data[0]
        else:
            return 0#TODO: change


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
    cp_agent = DQNHandler(env.observation_space, env.action_space, [4, 10, 10, 2, 'MSE', 'SGD', 0.01])
    avg_reward, win_streak = (0, 0)
    rewards = []
    losses = []
    exp = []
    exit = False
    count = 0
    for episode in range(1500):
        state = env.reset()

        # Reset the agent, if desired.
        cp_agent.reset()
        episode_reward = 0
        exp.append(cp_agent.explRate)
        print(count)

        # The total number of steps is limited (to avoid long-running loops).
        for steps in range(MAX_STEPS):
            env.render()
            count += 1
            # Ask the agent for the next action and step accordingly.
            # action = cp_agent.action(state)
            action = cp_agent.action(state)
            # action=0
            # if episode>30:
            #     if episode==31:
            #         state_next=state
            #     action=int((np.sign(state_next[3] + state_next[2]) + 1) / 2)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward

            # print(episode, ". ",state_next, reward, terminal, info)

            # Update any information inside the agent, if desired.

            losses.append(cp_agent.update(state, action, reward, state_next, terminal))
            # print(losses[-1])
            if losses[-1] > 1000000:
                print("high")
                # exit=True
                break

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

        if exit:
            break
        # print(rewards)

        # Has the agent succeeded?
        if win_streak == SUCCESS_STREAK and avg_reward >= SUCCESS_REWARD:
            return episode + 1, avg_reward
        print(losses[-1])

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

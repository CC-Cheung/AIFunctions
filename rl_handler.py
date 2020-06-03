from nn_handler import NNHandler
def accumMult(a):
    total=1
    for i in a:
        total*=i
    return total
#TODO: Add memory (tensor state, reward, next, number for action), Add adding obs, override load data with take from memory, override train
class DQNHandler(NNHandler):
    def __init__(self, observation_space, action_space, MLPDesc, batchSize=64, gamma=0.995,lr=0.007, maxMemSize=10000):
        super().__init__()
        self.batchSize=batchSize
        self.gamma=gamma
        self.maxMemSize=maxMemSize
        #uneeded
        self.inSize = accumMult(observation_space.shape)
        assert self.inSize==MLPDesc[0], "insize {} !=observation total dimension {}".format(self.inSize,MLPDesc[0])
        self.observation_space=observation_space

        self.outSize = action_space.n
        assert self.outSize == MLPDesc[-2], "outsize {} !=action total dimension {}".format(self.outSize, MLPDesc[-2])
        self.action_space=action_space

        self.model = NNHandler()
        self.lr=lr
        self.curLoss=0

        self.memory = [None for i in range(maxMemSize)]
        self.tData = np.zeros([maxMemSize, self.inSize])
        self.memCount=0
        self.memEnd=0
        self.stop=False
    def train(self,numStep):
        tData=self.tData[0:self.memEnd]

        indices = np.random.permutation(len(tData))
        excerpt = indices[0:self.batchSize]
        tData=tData[excerpt]

        tValues=self.predict(tData)
        for i,ex in enumerate(excerpt):
            state, action, reward, nextState, terminal = self.memory[ex]
            if terminal:
                tValues[i, action]=reward
            else:
                tValues[i, action] = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(np.array(nextState), axis=0)))
        # self.model.printIfNan()
        if self.stop:
            return -1
        temp= self.model.train(tData, tValues)
        # print(numStep, "temp=",temp)
        if numStep>420:
            self.stop=True
            return temp
        if temp>1000 or temp!=temp:
            print("high loss")
            self.model.highLossMode=True
            self.model.highLossWbAdjust(self.lr*2/3)
        else:
            self.model.highLossMode=False
            self.model.setlr(self.lr)
        return temp

    def predict(self, state):
        return self.model.predict(state)
    def update(self, state, action, reward, nextState, terminal,numStep):
        temp=np.random.random()

        index = self.memCount % self.maxMemSize
        if terminal:
            nextState=None #not needed
            if numStep > 500:
                reward = reward * -5
            else:
                reward = reward * 5

        self.memory[index] = (state, action, reward, nextState, terminal)
        self.tData[index, :] = np.expand_dims(np.array(state), axis=0)
        self.memCount+=1

        if self.memEnd<self.maxMemSize:
            self.memEnd+=1

        if self.memEnd < self.batchSize or temp > 0.25:
            return -1
        self.curLoss = self.train(numStep)
        return self.curLoss
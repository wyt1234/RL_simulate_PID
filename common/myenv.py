import numpy as np


class Space():
    def __init__(self, shape):
        self.shape = (shape,)


class MyEnv():
    def __init__(self, ):
        self.observation_space = Space(6)
        self.action_space = Space(2)
        self.reward = 0

    def seed(self, seed):
        np.random.seed(seed)

    def sample(self):
        a = np.random.uniform(low=-1, high=1, size=(1, 3))[0]
        return a * self.action_bound / 5

    # 如何定义奖励
    def get_reward(self):
        pass

    def step(self, action):
        # TODO 每个回合的步骤是否超过阈值，判断是否结束
        pass

    def reset(self):  # 重置环境
        pass

    def close(self):
        return

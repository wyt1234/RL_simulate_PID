import numpy as np


class MyEnv():
    def __init__(self):
        observation_space = None
        action_space = None

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        pass

    def reset(self):  # 重置环境
        pass

    def close(self):
        return

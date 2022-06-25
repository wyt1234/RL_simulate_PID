import numpy as np
import torch

from rotor_state_calculator import *


class Space():
    def __init__(self, shape):
        self.shape = (shape,)


class MyEnv():
    def __init__(self, ):
        self.reset()

    # 重置环境
    def reset(self):
        self.observation_space = Space(6)
        self.action_space = Space(2)
        self.reward = 0
        self.reward_rate = 10  # 奖励10倍于距离缩小
        # fixme 设定一个最终状态（悬停目标）
        self.final_state = np.array([1.4, .0, .0, 0.45, .0, .0])  # 最终需要的观测值
        self.last_observation = np.zeros((len(self.final_state)))  # 最近一次的观测 -> 初始值是0，可修改
        self.last_distance = self.o_distance(self.final_state, self.last_observation)  # 初始化距离比较量
        self.total_step = 1

    # 欧式距离
    def o_distance(self, vector1, vector2):
        return np.sqrt(np.sum(np.square(vector1 - vector2)))

    def seed(self, seed):
        np.random.seed(seed)

    # def sample(self):
    #     a = np.random.uniform(low=-1, high=1, size=(1, 3))[0]
    #     return a * self.action_bound / 5

    # 回合是否或者
    def is_alive(self, state: np.array):
        return np.max(state) >= 15 or np.min(state) <= -15

    # 定义奖励：目前状态距目标的距离与前一次状态距目标的距离的差值按比例缩放
    def get_reward(self, state: np.array):
        distance = self.o_distance(state, self.final_state)
        self.reward = self.reward_rate * (distance - self.last_distance)
        return self.reward

    def step(self, action: torch.Tensor):
        # TODO 每个回合的步骤是否超过阈值，判断是否结束
        self.last_observation = device_next_state(self.last_observation, action.tolist())
        reward = self.get_reward(self.last_observation)
        done = (not self.is_alive(self.last_observation)) or (self.total_step >= 1200)
        self.total_step += 1
        return self.last_observation, reward, done, 0

    def close(self):
        return


if __name__ == '__main__':
    env = MyEnv()
    print(env.last_distance)

import time
import dqn
import gym


import models

env = gym.make("LunarLander-v2")


class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward / 10.0

        if reward < -1.0:
            reward = -1.0

        if reward > 1.0:
            reward = 1.0

        return obs, reward, [done, done], info

env = SetRewardRange(env)
env.reset()

obs             = env.observation_space.shape
actions_count   = env.action_space.n

    
model = models.ModelDQN(obs, actions_count, [64, 64])
agent = dqn.Agent(env, model)


for iteration in range(1000000):
    agent.main()

    if iteration%256 == 0:
        print("iterations = ", iteration, " score = ", agent.score)
        env.render()

while True:
    agent.main()
    env.render()
    time.sleep(0.01)

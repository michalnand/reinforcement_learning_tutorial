import time
import dqn
import gym


import models

gym.envs.register(
    id='MountainCarCustom-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=4096      # MountainCar-v0 uses 200
)

env = gym.make("MountainCarCustom-v0")


class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward < 0:
            reward = -0.001

        if done: 
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

import time
import gym


from agents.agent_dqn  import *
from models.model_dqn  import *


#environment wrapper, reward scaling
class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward / 100.0

        return obs, reward, done, info

#create environment
env = gym.make("LunarLander-v2")
env = SetRewardRange(env)
env.reset()


#create DQN agent
agent = AgentDQN(env, ModelDQN)

'''
#train
for iteration in range(500000):
    agent.main()

    if iteration%256 == 0:
        print("iterations = ", iteration, " score = ", agent.score_episode)
        env.render()

#save model
agent.save("./models/")
'''

#load model
agent.load("./models/")
agent.epsilon = 0.1

#show how's running
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
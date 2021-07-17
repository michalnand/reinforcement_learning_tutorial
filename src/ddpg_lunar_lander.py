import time
import gym


from agents.agent_ddpg                      import *
from models.lunar_lander_model_ddpg         import *


#environment wrapper, reward scaling
class SetRewardRange(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward / 100.0

        return obs, reward, done, info

#create environment
env = gym.make("LunarLanderContinuous-v2")
env = SetRewardRange(env)
env.reset()

#create DDPG agent
agent = AgentDDPG(env, ModelActor, ModelCritic)

'''
#train, uncomment for run training
for iteration in range(1000000):
    agent.main()

    if iteration%256 == 0:
        print("iterations = ", iteration, " score = ", agent.score_episode)
        env.render()

#save model
agent.save("./models/trained/lunar_lander_")
'''

#load model
agent.load("./models/trained/lunar_lander_")
agent.epsilon = 0.2

#show how's running
while True:
    agent.main()
    env.render()
    time.sleep(0.01)

import time
import numpy
import gym
import gym_line_follower


from agents.agent_ddpg                  import *
from models.line_follower_model_ddpg    import *

#environment wrapper, state shaping
class Wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space.shape = (8, 2)

    def reset(self):
        obs = self.env.reset()
        return self._state(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = self._state(obs)

        return obs, reward, done, info

    def _state(self, obs):
        obs = numpy.array(obs, dtype=numpy.float32).reshape(8, 2)
        return obs

#create environment
env = gym.make("LineFollower-v0", nb_cam_pts=8, gui=False)
env = Wrapper(env)
obs = env.reset()


#create DDPG agent
agent = AgentDDPG(env, ModelActor, ModelCritic)


#train, uncomment for run training
for iteration in range(1000000):
    agent.main()

    if iteration%256 == 0:
        print("iterations = ", iteration, " score = ", agent.score_episode)

#save model
agent.save("./models/line_follower_")

'''
#load model
agent.load("./models/line_follower_")
agent.epsilon = 0.2

#show how's running
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
'''
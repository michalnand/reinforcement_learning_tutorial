import gym
import gym_anytrading

from agents.agent_dqn               import *
from models.stock_model_dqn_seq     import *

#create environment
env = gym.make("forex-v0")
env.reset()

#create DQN agent
agent = AgentDQN(env, ModelDQNSeq)

#train, uncomment for run training
for iteration in range(1000000):
    agent.main()

    if iteration%256 == 0:
        print("iterations = ", iteration, " score = ", agent.score_episode)
        env.render()

import numpy
import torch


class AgentDDPG():
    def __init__(   self, env,  ModelActor, ModelCritic, 
                    learning_rate_actor     = 0.0001,
                    learning_rate_critic    = 0.0002,
                    update_frequency        = 4,
                    tau                     = 0.001,
                    batch_size              = 32,

                    epsilon_decay       = 0.99999,
                    epsilon_start       = 1.0,
                    epsilon_end         = 0.1,

                    gamma                   = 0.99,
                    replay_buffer_size      = 8192 ):
       

        self.env    = env

        self.update_frequency = update_frequency
        self.tau            = tau
        self.batch_size     = batch_size

        self.epsilon_decay  = epsilon_decay
        self.epsilon        = epsilon_start
        self.epsilon_end    = epsilon_end
        
        self.gamma          = gamma

        self.replay_buffer_size = replay_buffer_size

        #obtain state shape, and actions count
        self.observation_shape  = self.env.observation_space.shape
        self.actions_count      = self.env.action_space.shape[0]

        #create replay buffer
        self.buffer_observation = numpy.zeros((self.replay_buffer_size, ) + self.observation_shape, dtype=numpy.float32)
        self.buffer_action      = numpy.zeros((self.replay_buffer_size, self.actions_count), dtype=numpy.float32)
        self.buffer_reward      = numpy.zeros((self.replay_buffer_size), dtype=numpy.float32)
        self.buffer_done        = numpy.zeros((self.replay_buffer_size), dtype=numpy.float32)
        
        #create models, actor, critic + its target networks, also copy parameters
        self.model_actor            = ModelActor(self.observation_shape, self.actions_count)
        self.model_actor_target     = ModelActor(self.observation_shape, self.actions_count)

        self.model_critic           = ModelCritic(self.observation_shape, self.actions_count)
        self.model_critic_target    = ModelCritic(self.observation_shape, self.actions_count)
 
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= learning_rate_actor)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= learning_rate_critic)


        #initial observation
        self.observation    = env.reset()
        
        self.iterations     = 0
        self.buffer_ptr     = 0

        self.score_sum      = 0
        self.score_episode  = 0

    
    def main(self):
        #epsilon decay
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon*self.epsilon_decay

        obs_t   = torch.from_numpy(self.observation).unsqueeze(0)
        
        #select action
        action  = self._sample_action(obs_t, self.epsilon)
        
        #environment step
        observation, reward, done, _ = self.env.step(action)

        #store transition to buffer
        self.buffer_observation[self.buffer_ptr]    = self.observation
        self.buffer_action[self.buffer_ptr]         = action
        self.buffer_reward[self.buffer_ptr]         = reward
        self.buffer_done[self.buffer_ptr]           = 1.0*done

        #move buffer pointer
        self.buffer_ptr = (self.buffer_ptr + 1)%self.replay_buffer_size

        #train model
        if self.iterations%self.update_frequency == 0:
            self.train_model()

        #update observation
        self.observation = observation
        
        self.score_sum+= reward
        if done:
            self.observation    = self.env.reset()

            #log smoothed score
            k = 0.02
            self.score_episode  = (1.0 - k)*self.score_episode + k*self.score_sum
            self.score_sum      = 0.0


        self.iterations+= 1
        
        
    def train_model(self):
        #obtain state, corresponding actions and rewards
        state_t, state_next_t, actions_t, rewards_t, dones_t = self._get_random_batch(self.batch_size)

        #predict next values
        action_next_t   = self.model_actor_target.forward(state_next_t).detach()
        value_next_t    = self.model_critic_target.forward(state_next_t, action_next_t).detach()

        #critic loss, Q-learning and MSE loss
        value_target    = rewards_t + self.gamma*dones_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, actions_t)

        critic_loss     = ((value_target - value_predicted)**2)
        critic_loss     = critic_loss.mean()
     
        #update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward() 
        self.optimizer_critic.step()

        #actor loss, maxime value
        actor_loss      = -self.model_critic.forward(state_t, self.model_actor.forward(state_t))
        actor_loss      = actor_loss.mean()

        #update actor
        self.optimizer_actor.zero_grad()       
        actor_loss.backward()
        self.optimizer_actor.step()


        #smooth update target networks 
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       
        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)


    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def _get_random_batch(self, batch_size):
        indices     = numpy.random.randint(0, self.replay_buffer_size - 1, size=batch_size)
        
        #sample batch
        states          = torch.from_numpy(numpy.take(self.buffer_observation,  indices, axis=0))
        states_next     = torch.from_numpy(numpy.take(self.buffer_observation,  indices+1, axis=0))
        actions         = torch.from_numpy(numpy.take(self.buffer_action,       indices, axis=0))
        rewards         = torch.from_numpy(numpy.take(self.buffer_reward,       indices, axis=0))
        dones           = torch.from_numpy(numpy.take(self.buffer_done,         indices, axis=0))

        return states, states_next, actions, rewards, dones


    def _sample_action(self, state_t, epsilon):
        #forward actor to obtain action
        action_t    = self.model_actor(state_t)

        #add noise to action
        action_t    = action_t + epsilon*torch.randn(action_t.shape)
        action_t    = action_t.clamp(-1.0, 1.0)

        #to numpy
        action      = action_t.squeeze(0).detach().to("cpu").numpy()

        return action


    def save(self, path = "./"):
        print("saving to", path)
        torch.save(self.model_actor.state_dict(), path + "model_ddpg_actor.pt")
        torch.save(self.model_critic.state_dict(), path + "model_ddpg_critic.pt")

    def load(self, path = "./"):
        print("loading from ", path)
        self.model_actor.load_state_dict(torch.load(path + "model_ddpg_actor.pt", map_location = "cpu"))
        self.model_critic.load_state_dict(torch.load(path + "model_ddpg_critic.pt", map_location = "cpu"))

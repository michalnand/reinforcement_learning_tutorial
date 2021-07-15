import numpy
import torch


class AgentDQN():
    def __init__(   self, env, Model, 
                    learning_rate       = 0.0001,
                    batch_size          = 32,
                    update_frequency    = 4,
                    update_model        = 4096,

                    epsilon_decay       = 0.99999,
                    epsilon_start       = 1.0,
                    epsilon_end         = 0.1,

                    gamma               = 0.99,
                    replay_buffer_size  = 8192 ):

        self.env    = env

        self.batch_size     = batch_size

        self.epsilon_decay  = epsilon_decay
        self.epsilon        = epsilon_start
        self.epsilon_end    = epsilon_end
        self.gamma          = gamma

        self.update_frequency   = update_frequency
        self.update_model       = update_model
        self.replay_buffer_size = replay_buffer_size

        #obtain state shape, and actions count
        self.observation_shape  = self.env.observation_space.shape
        self.actions_count      = self.env.action_space.n

        #create replay buffer
        self.buffer_observation = numpy.zeros((self.replay_buffer_size, ) + self.observation_shape, dtype=numpy.float32)
        self.buffer_action      = numpy.zeros((self.replay_buffer_size), dtype = int)
        self.buffer_reward      = numpy.zeros((self.replay_buffer_size), dtype=numpy.float32)
        self.buffer_done        = numpy.zeros((self.replay_buffer_size), dtype=numpy.float32)
        

        #create model
        self.model              = Model(self.observation_shape, self.actions_count)
        self.model_target       = Model(self.observation_shape, self.actions_count)
        self.optimizer          = torch.optim.Adam(self.model.parameters(), lr= learning_rate)

        #copy model to model target = make same models
        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)

        #initial observation
        self.observation    = env.reset()
        
        self.iterations = 0
        self.buffer_ptr = 0

        self.score_sum = 0
        self.score_episode = 0

    
    def main(self):
        #epsilon decay
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon*self.epsilon_decay
        
        #obtain q-values
        obs_t       = torch.from_numpy(self.observation).unsqueeze(0)
        q_values    = self.model(obs_t)
        q_values    = q_values.squeeze(0).detach().to("cpu").numpy()

        #select action
        action      = self._sample_action(q_values, self.epsilon)
        
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

        #copy to target model
        if self.iterations%self.update_model == 0:
            self.model_target.load_state_dict(self.model.state_dict())

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

        #obtain model output q-values
        q_values      = self.model.forward(state_t)
        q_values_next = self.model_target.forward(state_next_t)

 
        #q-learning equation
        q_target    = q_values.clone()

        q_max, _    = torch.max(q_values_next, axis=1)
        q_new       = rewards_t + self.gamma*(1.0 - dones_t)*q_max
        q_target[range(self.batch_size), actions_t] = q_new

        #MSE loss
        self.optimizer.zero_grad()
        loss = ((q_target.detach() - q_values)**2).mean() 
        loss.backward()
        self.optimizer.step()

    #e-greedy action selecting
    def _sample_action(self, q_values, epsilon):
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



    def save(self, path = "./"):
        print("saving to", path)
        torch.save(self.model.state_dict(), path + "model_dqn.pt")

    def load(self, path = "./"):
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "model_dqn.pt", map_location = "cpu"))
   
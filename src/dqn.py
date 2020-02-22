import numpy
import torch


class Agent():
    def __init__(self, env, model, 
                    learning_rate       = 0.0001,
                    batch_size          = 32,
                    update_frequency    = 32,


                    epsilon_decay   = 0.99999,
                    epsilon_start   = 1.0,
                    epsilon_end     = 0.1,

                    gamma               = 0.999,
                    replay_buffer_size  = 8192 ):

        self.env    = env

        self.batch_size     = batch_size

        self.epsilon_decay  = epsilon_decay
        self.epsilon        = epsilon_start
        self.epsilon_end    = epsilon_end
        self.gamma          = gamma

        self.update_frequency   = update_frequency
        self.replay_buffer_size = replay_buffer_size

        self.observation_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        shape = (self.replay_buffer_size, ) + self.observation_shape
        self.buffer_observation = numpy.zeros(shape)
        self.buffer_q_values    = numpy.zeros((self.replay_buffer_size, self.actions_count))
        self.buffer_action      = numpy.zeros((self.replay_buffer_size), dtype = int)
        self.buffer_reward      = numpy.zeros((self.replay_buffer_size))
        self.buffer_done        = numpy.zeros((self.replay_buffer_size), dtype = bool)
        

        self.model      = model

        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr= learning_rate)

        self.observation    = env.reset()
        
        self.iterations = 0
        self.score = 0
        self.buffer_ptr = 0

    
    def main(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon*self.epsilon_decay
        
        q_values = self.model.get_q_values(self.observation)
        action = self.choose_action_e_greedy(q_values, self.epsilon)
        

        observation, reward, done, _ = self.env.step(action)

        round_done = done[0]
        game_done  = done[1]

        
        self.buffer_observation[self.buffer_ptr]    = self.observation
        self.buffer_q_values[self.buffer_ptr]       = q_values
        self.buffer_action[self.buffer_ptr]         = action
        self.buffer_reward[self.buffer_ptr]         = reward
        self.buffer_done[self.buffer_ptr]           = round_done

        self.buffer_ptr = (self.buffer_ptr + 1)%self.replay_buffer_size

        if self.iterations%self.update_frequency == 0:
            self.train_model()

        self.observation = observation
            
        if game_done:
            self.env.reset()

        self.iterations+= 1
        self.score+= reward
        
        
    def train_model(self):
        input, q_target = self._get_random_batch(self.batch_size)


        q_predicted = self.model.forward(input)

        self.optimizer.zero_grad()

        loss = ((q_target - q_predicted)**2).mean() 
        loss.backward()
        self.optimizer.step()

    def choose_action_e_greedy(self, q_values, epsilon):
        result = numpy.argmax(q_values)
        
        if numpy.random.random() < epsilon:
            result = numpy.random.randint(len(q_values))
        
        return result

    def _get_random_batch(self, batch_size):
        
        state_shape   = (batch_size, ) + self.observation_shape[0:]
        q_values_shape = (batch_size, ) + (self.actions_count, )

        input       = torch.zeros(state_shape,  dtype=torch.float32)
        target      = torch.zeros(q_values_shape,  dtype=torch.float32)
 
        for i in range(0, batch_size): 
            n      = numpy.random.randint(self.replay_buffer_size - 1)

            if self.buffer_done[n]:
                gamma_ = 0.0
            else: 
                gamma_ = self.gamma
    
            q_values    = self.buffer_q_values[n].copy()
            action      = self.buffer_action[n]

            q_values[action] = self.buffer_reward[n] + gamma_*numpy.max(self.buffer_q_values[n+1])
            
            input[i]  = torch.from_numpy(self.buffer_observation[n])
            target[i] = torch.from_numpy(q_values)
            
        return input, target

        
# reinforcement learning tutorial

short tutorial for basic deep reinforcement learning

- minimalistic implementation of DQN and DDPG

![animation](doc/lunar_lander.gif)

## dqn - discrete action space

![dqn](doc/dqn.png)

### model architecture

- 3 layers
- ReLU activation
- xavier weight init, zero bias

![model_dqn](doc/modeldqn.png)



## ddpg - continuous action space

![dqn](doc/ddpg.png)

### model architecture

- 3 layers 
- ReLU activation
- xavier weight init, zero bias
- actor output layer init range <-0.3, 0.3>, with tanh
- critic output layer init range <-0.003, 0.003>

![dqn](doc/modelddpg.png)

# reinforcement learning tutorial

short tutorial for basic deep reinforcement learning

- minimalistic implementation of DQN and DDPG

![animation](doc/lunar_lander.gif)

## install and run 

### dependences : 
```bash
pip3 install -r requirements.txt
```


### run pretrained agent : 
```bash
cd src
python3 dqn_lunar_lander.py
```

### for training from scratch uncomment in dqn_lunar_lander.py and run:
```python
#train
for iteration in range(500000):
    agent.main()

    if iteration%256 == 0:
        print("iterations = ", iteration, " score = ", agent.score_episode)
        env.render()

#save model
agent.save("./models/")
```


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


## TODO 

- DDPG still not running
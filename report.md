## Learning Algorithm
Four DQN algorithms were implemented and trained.

### Vanilla DQN

- Fully-connected layer - input: 37 (state size) output: 64
- Fully-connected layer - input: 64 output 64
- Fully-connected layer - input: 64 output: (action size)

- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.999

### Double DQN

- Fully-connected layer - input: 37 (state size) output: 64
- Fully-connected layer - input: 64 output 64
- Fully-connected layer - input: 64 output: (action size)

- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.999


### Dueling DQN

- Fully-connected layer - input: 37 (state size) output: 64

- Fully-connected layer
    - input: 64 output 64 for approximation state-dependent action advantage function
    - input: 64 output 64 for approximation state value function
- Fully-connected layer
    - input: 64 output: (action size)  for approximation state-dependent action advantage function
    - input: 64 output: 1  for approximation state value function

- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.999

### Double Dueling DQN

- Fully-connected layer - input: 37 (state size) output: 64

- Fully-connected layer
    - input: 64 output 64 for approximation state-dependent action advantage function
    - input: 64 output 64 for approximation state value function
- Fully-connected layer
    - input: 64 output: (action size)  for approximation state-dependent action advantage function
    - input: 64 output: 1  for approximation state value function

- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.999

## Why DQN works
A Deep Q Network learns by using a neural network to approximate the action value function which is inturned used to achieve the optimal policy which guides the agent to perform the optimal action in a given state to earn the maximum reward possible at that state. 

The technology works really well in this scenario as this is an episodic task with a limited state and action space. Since the possibile state-action pairs is relatively low in comparison to more complex problems, the agent can easily explore the given environment during the exploration phase while learning the potential rewards attained for prospective actions and exploit this knowledge later on while it learns to get better at maximizing the potential rewards that it can earn.


## Plot of Rewards
### Vanilla DQN
Episode 100	Average Score: 0.53
Episode 200	Average Score: 3.17
Episode 300	Average Score: 6.48
Episode 400	Average Score: 10.44
Episode 500	Average Score: 12.20
Episode 531	Average Score: 13.01
Environment solved in 431 episodes!	Average Score: 13.01

### Dueling DQN
Episode 100	Average Score: 0.88
Episode 200	Average Score: 4.87
Episode 300	Average Score: 8.23
Episode 400	Average Score: 10.16
Episode 500	Average Score: 11.78
Episode 561	Average Score: 13.04
Environment solved in 461 episodes!	Average Score: 13.04

### Double DQN
Episode 100	Average Score: 0.42
Episode 200	Average Score: 3.99
Episode 300	Average Score: 7.58
Episode 400	Average Score: 9.81
Episode 500	Average Score: 12.56
Episode 510	Average Score: 13.00
Environment solved in 410 episodes!	Average Score: 13.00

### Double Dueling DQN
Episode 100	Average Score: 0.72
Episode 200	Average Score: 4.40
Episode 300	Average Score: 8.20
Episode 400	Average Score: 11.01
Episode 468	Average Score: 13.03
Environment solved in 368 episodes!	Average Score: 13.03

## Ideas for Future Improvements
- Implementation of Prioritized Experience Replay
- Implementation Rainbow
- Implementation of Policy or Actor Critic based methods
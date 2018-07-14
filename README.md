# Solving Connect 4 using DQN [WIP]

The goal is to solve Connect4 in an unsupervised fashion. The work is still in progress. Below is the list of methods and architectures currently in use to try to make it work.

- Artificial Neural Network vs [MiniMax bot](http://connect4.gamesolver.org/?pos=)
- Q value approximation with the ANN
- e-greedy Boltzmann exploration
- Prioritised Experience Replay using a [SumTree](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/)
- Error clipping using logcosh and Huber loss functions
- Reward clipping to [-1, 1]
- Double DQN with target network update

# Deep-Reinforcement-Learning
Deep Reinforcement Learning Algorithms and Code - Explanations of research papers and their implementations

1. `REINFORCE`: Vanilla Policy Gradient
2. `DQN`: Deep Q-Learning
3. `A3C/A2C`: Asynchronous methods for Deep RL
4. `PPO`: Proximal Policy Optimization
5. `DDPG`: Deep Deterministic Policy Gradient

(`DeepRL.md`: General tips on Deep reinforcement Learning)


From Open AI ["Spinning Up as a Deep RL Researcher"](https://spinningup.openai.com/en/latest/spinningup/spinningup.html):

>**Which algorithms?** *You should probably start with vanilla policy gradient (also called REINFORCE), DQN, A2C (the synchronous version of A3C), PPO (the variant with the clipped objective), and DDPG, approximately in that order. The simplest versions of all of these can be written in just a few hundred lines of code (ballpark 250-300), and some of them even less (for example, a no-frills version of VPG can be written in about 80 lines). Write single-threaded code before you try writing parallelized versions of these algorithms. (Do try to parallelize at least one.)*



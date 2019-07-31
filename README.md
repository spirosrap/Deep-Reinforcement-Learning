# Deep-Reinforcement-Learning
Deep Reinforcement Learning Algorithms and Code - Explanations of research papers and their implementations (*All algorithm implementations are done in Pytorch*)

1. `REINFORCE`: Vanilla Policy Gradient
2. `DQN`: [Deep Q-Learning, Mnih et al, 2013](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
3. `A3C/A2C`: [Asynchronous methods for Deep RL,Mnih et al, 2016  ](https://arxiv.org/abs/1602.01783)
4. `PPO`: [Proximal Policy Optimization,Schulman et al, 2017](https://arxiv.org/abs/1707.06347)
5. `DDPG`: [Deep Deterministic Policy Gradient,Lillicrap et al, 2015](https://arxiv.org/abs/1509.02971)

(`DeepRL.md`: General tips on Deep reinforcement Learning)


**From Open AI ["Spinning Up as a Deep RL Researcher"](https://spinningup.openai.com/en/latest/spinningup/spinningup.html).**: How to start in Deep RL assuming you've got a solid background in Mathematics([1](http://wiki.fast.ai/index.php/Calculus_for_Deep_Learning),[2](https://www.quantstart.com/articles/matrix-algebra-linear-algebra-for-deep-learning-part-2)), a general knowledge of Deep Learning and are familiar with at least one Deep Learning Library (Like [PyTorch](https://pytorch.org/)  or [TensorFlow](https://www.tensorflow.org/)):

![OPEN AI](https://spinningup.openai.com/en/latest/_static/spinning-up-logo2.png)

>**Which algorithms?** *You should probably start with vanilla policy gradient (also called REINFORCE), DQN, A2C (the synchronous version of A3C), PPO (the variant with the clipped objective), and DDPG, approximately in that order. The simplest versions of all of these can be written in just a few hundred lines of code (ballpark 250-300), and some of them even less (for example, a no-frills version of VPG can be written in about 80 lines). Write single-threaded code before you try writing parallelized versions of these algorithms. (Do try to parallelize at least one.)*



Further Algorithms to study (Open AI Hackathon):

* `TRPO`: [Schulman et al, 2015](https://arxiv.org/abs/1502.05477)
* `C51`: [Bellemare et al, 2017](https://arxiv.org/abs/1707.06887)
* `QR-DQN`: [Dabney et al, 2017](https://arxiv.org/abs/1710.10044)
* `SVG`: [Heess et al, 2015](https://arxiv.org/abs/1510.09142)
* `I2A`: [Weber et al, 2017](https://arxiv.org/abs/1707.06203)
* `MBMF`: [Nagabandi et al, 2017](https://sites.google.com/view/mbmf)
* `AlphaZero`: [Silver et al, 2017](https://arxiv.org/abs/1712.01815)

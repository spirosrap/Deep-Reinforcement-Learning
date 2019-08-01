[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


# Project 1: Navigation: The Banana Environment - Learn from pixels

*Environment taken [from Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation))*


### Introduction

Train an agent to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

learn directly from pixels!

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state is an 84 x 84 RGB image, corresponding to the agent's first-person view.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Download the environment

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Headless Linux [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux_NoVis.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in your main directory, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.

### Getting Started


### Instructions

Follow the instructions in `Navigation_Pixels.ipynb` to get started with training your own agent!

### Description of the code

To train the agent in the environment I used the Double Deep Q Learning algorithm. This is an extension of the Deep Q Learning algorithm which I'll explain shortly. The Deep Q learning algorithm is in turn an extension of the classical Q-learning algorithm of reinforcement Learning. In the Q learning algorithm we're trying to estimate the optimal policy by first trying to estimate the optimal action value function $Q_{\pi}(s,α)$. In the Deep Q-learning algorithm we represent the action value function as a neural network instead of a table. The Double Q-learning algorithm is an extension of the Deep Q learning algorithm in which we differentiate on how to approximate the action value function. Specifically, we select the best action using on set of parameters $w$ but we evaluate the action on a different set of parameters $w^\prime$:

$$
R+\gamma \hat{q}\left(S^{\prime}, \arg \max _{a} \hat{q}\left(S^{\prime}, a, \mathbf{w}\right), \mathbf{w}^{\prime}\right)
$$

For the model I used a simple neural network with three fully connected layers (No CNNs).

I have also tried to apply another optimization of the algorithm called experience learning (also described in the lessons) but I didn't get good results so, I've skipped it from my submission.You can find the implementation on the file `DDQN_exp.py`.

### Ideas for Future Work

* Further optimize the model and the agent so that the experience replay improvement gives better results
* Try with different model architectures. For example, use of a convolutional layer or more fully connected layers.
* More parameter tuning
